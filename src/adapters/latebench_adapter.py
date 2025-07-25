"""
LateBench Format Adapter for Seamless Integration

This module provides adapters and utilities for seamless integration between
different data formats in the LateBench ecosystem, ensuring compatibility
across evaluation, storage, and dashboard systems.
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import logging
from pathlib import Path

from src.critic import CriticResult, LLMCritic
from src.critic_batch import BatchCriticEvaluator, BatchEvaluationConfig, EvaluationResult
from src.metrics.deltabench import DeltaBenchEvaluator, DeltaBenchMetrics
from src.storage.critic_store import CriticResultStore, StoredResult, BatchResultSummary
from src.dataset_manager import LateBenchDatasetManager
from src.data_processing.unified_schema import LateBenchExample

logger = logging.getLogger(__name__)


@dataclass
class EvaluationPipeline:
    """Configuration for end-to-end evaluation pipeline"""
    dataset_name: str
    model_version: str = "gpt-4o-mini"
    problem_type: str = "all"
    use_caching: bool = True
    compute_deltabench_metrics: bool = True
    use_first_error_cutoff: bool = True
    save_results: bool = True
    max_concurrent: int = 10
    rate_limit: int = 100
    
    # Callbacks
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None


class LateBenchAdapter:
    """Main adapter class for LateBench ecosystem integration"""
    
    def __init__(self, 
                 storage_dir: str = "./data/critic_store",
                 enable_dashboard_integration: bool = True):
        
        self.dataset_manager = LateBenchDatasetManager()
        self.critic_store = CriticResultStore(storage_dir)
        self.enable_dashboard_integration = enable_dashboard_integration
        
        # Cache for loaded datasets
        self._dataset_cache: Dict[str, List[LateBenchExample]] = {}
        self._deltabench_evaluator = DeltaBenchEvaluator()
        
    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available datasets"""
        return self.dataset_manager.list_available_datasets()
    
    def get_dataset_examples(self, dataset_name: str, problem_type: str = "all") -> List[LateBenchExample]:
        """Get examples from a dataset with caching"""
        cache_key = f"{dataset_name}_{problem_type}"
        
        if cache_key not in self._dataset_cache:
            if not self.dataset_manager.load_dataset(dataset_name, problem_type):
                raise ValueError(f"Failed to load dataset: {dataset_name}")
            
            examples = self.dataset_manager.get_current_examples()
            self._dataset_cache[cache_key] = examples
            logger.info(f"Loaded {len(examples)} examples from {dataset_name}")
        
        return self._dataset_cache[cache_key]
    
    def get_existing_results(self, 
                           dataset_name: str, 
                           model_version: Optional[str] = None) -> Dict[str, StoredResult]:
        """Get existing critic results for a dataset"""
        return self.critic_store.get_batch_results(dataset_name, model_version)
    
    def evaluate_dataset(self, pipeline: EvaluationPipeline) -> BatchResultSummary:
        """Run complete evaluation pipeline on a dataset"""
        logger.info(f"Starting evaluation pipeline for {pipeline.dataset_name}")
        
        # Load dataset
        examples = self.get_dataset_examples(pipeline.dataset_name, pipeline.problem_type)
        logger.info(f"Loaded {len(examples)} examples")
        
        # Check for existing results if caching enabled
        existing_results = {}
        if pipeline.use_caching:
            existing_results = self.get_existing_results(pipeline.dataset_name, pipeline.model_version)
            logger.info(f"Found {len(existing_results)} existing results")
        
        # Filter examples that need evaluation
        examples_to_evaluate = [
            ex for ex in examples 
            if not pipeline.use_caching or ex.id not in existing_results
        ]
        
        logger.info(f"Need to evaluate {len(examples_to_evaluate)} examples")
        
        # Configure batch evaluator
        config = BatchEvaluationConfig(
            model=pipeline.model_version,
            max_concurrent=pipeline.max_concurrent,
            rate_limit_per_minute=pipeline.rate_limit,
            enable_caching=pipeline.use_caching
        )
        
        batch_evaluator = BatchCriticEvaluator(config)
        
        # Add progress callback if provided
        if pipeline.progress_callback:
            batch_evaluator.add_progress_callback(pipeline.progress_callback)
        
        # Run batch evaluation
        new_results = asyncio.run(batch_evaluator.evaluate_dataset(examples_to_evaluate))
        
        # Combine with existing results
        all_critic_results = {}
        
        # Add existing results
        for example_id, stored_result in existing_results.items():
            all_critic_results[example_id] = stored_result.critic_result
        
        # Add new results
        for example_id, eval_result in new_results.items():
            if eval_result.critic_result:
                all_critic_results[example_id] = eval_result.critic_result
        
        # Compute DeltaBench metrics if requested
        deltabench_metrics = None
        if pipeline.compute_deltabench_metrics:
            logger.info("Computing DeltaBench metrics...")
            try:
                deltabench_metrics = self._deltabench_evaluator.evaluate_batch(
                    examples, all_critic_results
                )
                logger.info(f"DeltaBench F1: {deltabench_metrics.step_f1:.3f}")
            except Exception as e:
                logger.error(f"Error computing DeltaBench metrics: {e}")
        
        # Store results if requested
        batch_summary = None
        if pipeline.save_results:
            logger.info("Storing results...")
            batch_summary = self.critic_store.store_batch_results(
                all_critic_results,
                pipeline.model_version,
                pipeline.dataset_name,
                deltabench_metrics
            )
            
            # Update dashboard integration if enabled
            if self.enable_dashboard_integration:
                self._update_dashboard_results(all_critic_results)
        
        # Call completion callback if provided
        if pipeline.completion_callback:
            pipeline.completion_callback(batch_summary, deltabench_metrics)
        
        logger.info(f"Evaluation pipeline completed for {pipeline.dataset_name}")
        return batch_summary
    
    def evaluate_single_example(self, 
                               example_id: str, 
                               dataset_name: str,
                               model_version: str = "gpt-4o-mini",
                               force_refresh: bool = False) -> Optional[CriticResult]:
        """Evaluate a single example (for dashboard integration)"""
        
        # Check cache first unless forced refresh
        if not force_refresh:
            stored_result = self.critic_store.get_result(example_id, model_version, dataset_name)
            if stored_result:
                logger.info(f"Using cached result for {example_id}")
                return stored_result.critic_result
        
        # Load dataset and find example
        examples = self.get_dataset_examples(dataset_name)
        example = next((ex for ex in examples if ex.id == example_id), None)
        
        if not example:
            logger.error(f"Example {example_id} not found in dataset {dataset_name}")
            return None
        
        # Evaluate single example
        try:
            critic = LLMCritic(model=model_version)
            problem = example.problem.statement
            steps = [step.content for step in example.solution.steps]
            
            critic_result = critic.evaluate_solution(problem, steps)
            
            # Store result
            self.critic_store.store_result(example_id, critic_result, model_version, dataset_name)
            
            # Update dashboard if enabled
            if self.enable_dashboard_integration:
                self._update_dashboard_results({example_id: critic_result})
            
            logger.info(f"Evaluated {example_id}: {'errors found' if critic_result.has_errors else 'no errors'}")
            return critic_result
            
        except Exception as e:
            logger.error(f"Error evaluating {example_id}: {e}")
            return None
    
    def get_evaluation_summary(self, dataset_name: str) -> Dict[str, Any]:
        """Get comprehensive evaluation summary for a dataset"""
        
        # Get batch summaries
        batch_summaries = self.critic_store.get_batch_summaries(dataset_name)
        
        # Get latest results
        latest_results = self.critic_store.get_batch_results(dataset_name)
        
        # Calculate aggregate statistics
        total_evaluated = len(latest_results)
        total_with_errors = sum(1 for r in latest_results.values() if r.critic_result.has_errors)
        
        # Get latest DeltaBench metrics
        latest_metrics = None
        if batch_summaries:
            latest_summary = batch_summaries[0]  # Most recent
            latest_metrics = latest_summary.deltabench_metrics
        
        return {
            'dataset_name': dataset_name,
            'total_examples_evaluated': total_evaluated,
            'examples_with_errors_found': total_with_errors,
            'error_detection_rate': total_with_errors / total_evaluated if total_evaluated > 0 else 0,
            'batch_evaluations': len(batch_summaries),
            'latest_deltabench_metrics': latest_metrics.to_dict() if latest_metrics else None,
            'latest_batch_summary': {
                'batch_id': latest_summary.batch_id,
                'evaluation_end': latest_summary.evaluation_end,
                'successful_evaluations': latest_summary.successful_evaluations,
                'model_version': latest_summary.model_version
            } if batch_summaries else None
        }
    
    def export_results(self, 
                      dataset_name: str, 
                      output_path: str,
                      format: str = "json",
                      include_deltabench_metrics: bool = True) -> str:
        """Export evaluation results in various formats"""
        
        # Get results
        results = self.critic_store.get_batch_results(dataset_name)
        batch_summaries = self.critic_store.get_batch_summaries(dataset_name, limit=1)
        
        if format.lower() == "json":
            return self._export_json(results, batch_summaries, output_path, include_deltabench_metrics)
        elif format.lower() == "csv":
            return self._export_csv(results, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_dataset_cache(self, dataset_name: Optional[str] = None):
        """Clear dataset cache"""
        if dataset_name:
            keys_to_remove = [k for k in self._dataset_cache.keys() if k.startswith(dataset_name)]
            for key in keys_to_remove:
                del self._dataset_cache[key]
        else:
            self._dataset_cache.clear()
        
        logger.info(f"Cleared dataset cache for {dataset_name or 'all datasets'}")
    
    def cleanup_old_results(self, days: int = 30):
        """Cleanup old results and backups"""
        self.critic_store.cleanup_old_backups()
        logger.info(f"Cleaned up results older than {days} days")
    
    def create_backup(self) -> str:
        """Create backup of all stored results"""
        return self.critic_store.create_backup()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        storage_stats = self.critic_store.get_storage_stats()
        available_datasets = self.list_available_datasets()
        
        return {
            'storage_stats': storage_stats,
            'available_datasets': {
                'count': len(available_datasets),
                'datasets': available_datasets
            },
            'cached_datasets': list(self._dataset_cache.keys()),
            'dashboard_integration_enabled': self.enable_dashboard_integration
        }
    
    # Private methods
    
    def _update_dashboard_results(self, critic_results: Dict[str, CriticResult]):
        """Update dashboard with new critic results"""
        try:
            # Import here to avoid circular imports
            from dashboard.utils import DashboardData
            
            # This would need to be implemented to update the dashboard's
            # stored critic results
            logger.debug(f"Updated dashboard with {len(critic_results)} results")
            
        except ImportError:
            logger.warning("Dashboard integration not available")
        except Exception as e:
            logger.warning(f"Error updating dashboard: {e}")
    
    def _export_json(self, results: Dict[str, StoredResult], 
                    batch_summaries: List[BatchResultSummary],
                    output_path: str, include_metrics: bool) -> str:
        """Export results to JSON format"""
        import json
        
        export_data = {
            'export_timestamp': f"{asyncio.get_event_loop().time()}",
            'total_results': len(results),
            'results': {
                result_id: {
                    'example_id': result.example_id,
                    'model_version': result.model_version,
                    'evaluation_timestamp': result.evaluation_timestamp,
                    'critic_result': result.critic_result.to_dict()
                }
                for result_id, result in results.items()
            }
        }
        
        if include_metrics and batch_summaries:
            latest_summary = batch_summaries[0]
            if latest_summary.deltabench_metrics:
                export_data['deltabench_metrics'] = latest_summary.deltabench_metrics.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        return output_path
    
    def _export_csv(self, results: Dict[str, StoredResult], output_path: str) -> str:
        """Export results to CSV format"""
        import csv
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'example_id', 'model_version', 'evaluation_timestamp',
                'has_errors', 'error_steps', 'processing_time'
            ])
            
            # Data
            for result in results.values():
                writer.writerow([
                    result.example_id,
                    result.model_version,
                    result.evaluation_timestamp,
                    result.critic_result.has_errors,
                    ','.join(map(str, result.critic_result.error_steps)),
                    result.critic_result.processing_time
                ])
        
        logger.info(f"Exported {len(results)} results to {output_path}")
        return output_path


# Convenience functions

def run_full_evaluation(dataset_name: str, 
                       model_version: str = "gpt-4o-mini",
                       max_concurrent: int = 10) -> BatchResultSummary:
    """Run complete evaluation pipeline with default settings"""
    
    adapter = LateBenchAdapter()
    
    pipeline = EvaluationPipeline(
        dataset_name=dataset_name,
        model_version=model_version,
        max_concurrent=max_concurrent,
        use_caching=True,
        compute_deltabench_metrics=True,
        save_results=True
    )
    
    return adapter.evaluate_dataset(pipeline)


def get_evaluation_summary(dataset_name: str) -> Dict[str, Any]:
    """Get evaluation summary for a dataset"""
    adapter = LateBenchAdapter()
    return adapter.get_evaluation_summary(dataset_name)


def evaluate_single(example_id: str, dataset_name: str, force_refresh: bool = False) -> Optional[CriticResult]:
    """Evaluate a single example"""
    adapter = LateBenchAdapter()
    return adapter.evaluate_single_example(example_id, dataset_name, force_refresh=force_refresh)


if __name__ == "__main__":
    # Example usage
    adapter = LateBenchAdapter()
    
    # List available datasets
    datasets = adapter.list_available_datasets()
    print("Available datasets:", datasets)
    
    # Get system stats
    stats = adapter.get_system_stats()
    print("System stats:", stats)