#!/usr/bin/env python3
"""
Large-scale error injection with robust checkpointing and progress tracking.
"""

import sys
sys.path.append('src')

import json
import time
import os
import asyncio
import concurrent.futures
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm
from error_injector import AdversarialErrorInjector, InjectionResult

class LargeScaleInjector:
    """Handles large-scale error injection with robust checkpointing."""
    
    def __init__(self, experiment_dir: str, max_workers: int = 5):
        self.experiment_dir = experiment_dir
        self.checkpoints_dir = f"{experiment_dir}/checkpoints"
        self.results_dir = f"{experiment_dir}/results"
        self.logs_dir = f"{experiment_dir}/logs"
        self.max_workers = max_workers
        
        # Initialize multiple injectors for parallel processing
        self.injectors = [AdversarialErrorInjector() for _ in range(max_workers)]
        
        # Setup logging
        self.log_file = f"{self.logs_dir}/injection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.log("🚀 Initialized Large-Scale Error Injector")
        self.log(f"   Experiment directory: {experiment_dir}")
        self.log(f"   Max parallel workers: {max_workers}")
        
    def log(self, message: str):
        """Log message to both console and file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\\n')
    
    def save_checkpoint(self, results: List[InjectionResult], batch_num: int, total_processed: int):
        """Save checkpoint with current progress."""
        checkpoint_data = {
            'batch_number': batch_num,
            'total_processed': total_processed,
            'timestamp': datetime.now().isoformat(),
            'results': self._serialize_results(results)
        }
        
        checkpoint_path = f"{self.checkpoints_dir}/checkpoint_batch_{batch_num:03d}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.log(f"💾 Saved checkpoint: batch {batch_num}, {total_processed} problems processed")
        return checkpoint_path
    
    def load_latest_checkpoint(self) -> tuple:
        """Load the most recent checkpoint if available."""
        checkpoint_files = [f for f in os.listdir(self.checkpoints_dir) if f.startswith('checkpoint_batch_')]
        
        if not checkpoint_files:
            self.log("📋 No existing checkpoints found, starting fresh")
            return [], 0, 0
        
        # Sort by batch number and get latest
        checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
        
        checkpoint_path = f"{self.checkpoints_dir}/{latest_checkpoint}"
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        results = self._deserialize_results(checkpoint_data['results'])
        batch_num = checkpoint_data['batch_number']
        total_processed = checkpoint_data['total_processed']
        
        self.log(f"🔄 Loaded checkpoint: batch {batch_num}, {total_processed} problems processed")
        return results, batch_num, total_processed
    
    def _serialize_results(self, results: List[InjectionResult]) -> List[Dict]:
        """Convert InjectionResult objects to serializable format."""
        serialized = []
        for result in results:
            serialized.append({
                'success': result.success,
                'original_problem': result.original_problem,
                'modified_solution': result.modified_solution,
                'error_analysis': result.error_analysis,
                'error_explanation': result.error_explanation,
                'metadata': result.metadata,
                'error_message': result.error_message
            })
        return serialized
    
    def _deserialize_results(self, data: List[Dict]) -> List[InjectionResult]:
        """Convert serialized data back to InjectionResult objects."""
        results = []
        for item in data:
            results.append(InjectionResult(
                success=item['success'],
                original_problem=item['original_problem'],
                modified_solution=item['modified_solution'],
                error_analysis=item['error_analysis'],
                error_explanation=item['error_explanation'],
                metadata=item['metadata'],
                error_message=item.get('error_message')
            ))
        return results
    
    def process_single_problem(self, problem_data: tuple) -> InjectionResult:
        """Process a single problem with error injection."""
        problem, error_type, worker_id = problem_data
        injector = self.injectors[worker_id % len(self.injectors)]
        
        try:
            result = injector.inject_error(problem, error_type_preference=error_type)
            return result
        except Exception as e:
            # Create failed result for exceptions
            return InjectionResult(
                success=False,
                original_problem=problem,
                modified_solution={},
                error_analysis={},
                error_explanation={},
                metadata={'exception': str(e), 'worker_id': worker_id},
                error_message=f"Exception during processing: {str(e)}"
            )
    
    def run_parallel_batch(self, batch_problems: List[Dict], error_types: List[str], 
                          start_idx: int) -> List[InjectionResult]:
        """Process a batch of problems in parallel."""
        
        # Prepare problem data with error types and worker IDs
        problem_data = []
        for i, problem in enumerate(batch_problems):
            error_type = error_types[(start_idx + i) % len(error_types)]
            worker_id = i % self.max_workers
            problem_data.append((problem, error_type, worker_id))
        
        # Process in parallel with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with tqdm(desc=f"Batch processing", total=len(batch_problems), 
                     leave=False, ncols=80) as pbar:
                
                # Submit all tasks
                future_to_data = {
                    executor.submit(self.process_single_problem, data): data 
                    for data in problem_data
                }
                
                results = []
                for future in concurrent.futures.as_completed(future_to_data):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    # Update progress bar description with success info
                    if result.success:
                        pbar.set_postfix(status="✅", refresh=False)
                    else:
                        pbar.set_postfix(status="❌", refresh=False)
        
        return results
    
    def run_batch_injection(self, problems: List[Dict], batch_size: int = 20, 
                          checkpoint_interval: int = 3, resume: bool = True):
        """Run error injection with parallel processing, batching and checkpointing."""
        
        total_problems = len(problems)
        self.log(f"🎯 Starting parallel batch injection for {total_problems} problems")
        self.log(f"   Batch size: {batch_size}")
        self.log(f"   Parallel workers: {self.max_workers}")
        self.log(f"   Checkpoint interval: {checkpoint_interval} batches")
        
        # Load existing progress if resuming
        if resume:
            all_results, start_batch, total_processed = self.load_latest_checkpoint()
            start_idx = total_processed
        else:
            all_results, start_batch, total_processed = [], 0, 0
            start_idx = 0
        
        if start_idx >= total_problems:
            self.log("✅ All problems already processed!")
            return all_results
        
        self.log(f"🔄 Resuming from problem {start_idx + 1}/{total_problems}")
        
        # Error type distribution for variety
        error_types = ['logical_error', 'incorrect_rules_properties', 'invalid_generalization', 
                      'assumption_error', 'misunderstanding_conditions']
        
        # Process problems in batches with overall progress bar
        batch_num = start_batch
        remaining_problems = total_problems - start_idx
        
        with tqdm(total=remaining_problems, desc="Overall Progress", unit="problems", 
                 position=0, ncols=100) as main_pbar:
            
            for i in range(start_idx, total_problems, batch_size):
                batch_num += 1
                end_idx = min(i + batch_size, total_problems)
                batch_problems = problems[i:end_idx]
                
                self.log(f"🔄 Processing batch {batch_num}: problems {i+1}-{end_idx}")
                batch_start_time = time.time()
                
                # Process batch in parallel
                batch_results = self.run_parallel_batch(batch_problems, error_types, i)
                
                # Add batch results to total
                all_results.extend(batch_results)
                total_processed = i + len(batch_problems)
                
                # Update main progress bar
                main_pbar.update(len(batch_problems))
                
                # Calculate batch statistics
                batch_time = time.time() - batch_start_time
                batch_success_rate = sum(1 for r in batch_results if r.success) / len(batch_results)
                overall_success_rate = sum(1 for r in all_results if r.success) / len(all_results)
                
                # Update progress bar with statistics
                main_pbar.set_postfix({
                    'Batch': f'{batch_num}',
                    'Success': f'{batch_success_rate:.1%}',
                    'Overall': f'{overall_success_rate:.1%}',
                    'Time': f'{batch_time:.1f}s'
                })
                
                self.log(f"📊 Batch {batch_num} complete:")
                self.log(f"   Time: {batch_time:.1f}s ({batch_time/len(batch_problems):.1f}s per problem)")
                self.log(f"   Batch success rate: {batch_success_rate:.1%}")
                self.log(f"   Overall success rate: {overall_success_rate:.1%}")
                self.log(f"   Total processed: {total_processed}/{total_problems}")
                
                # Save checkpoint
                if batch_num % checkpoint_interval == 0 or total_processed >= total_problems:
                    self.save_checkpoint(all_results, batch_num, total_processed)
        
        # Save final results
        final_path = f"{self.results_dir}/final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.injectors[0].save_results(all_results, final_path)
        
        # Generate summary
        self.generate_summary(all_results, total_problems)
        
        return all_results
    
    def generate_summary(self, results: List[InjectionResult], total_problems: int):
        """Generate and save experiment summary."""
        
        successful_results = [r for r in results if r.success]
        success_rate = len(successful_results) / len(results) if results else 0
        
        # Error type distribution
        error_types = {}
        for result in successful_results:
            error_type = result.error_analysis.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Step placement analysis
        step_placements = []
        for result in successful_results:
            error_step = result.error_analysis.get('selected_error_step')
            total_steps = result.metadata.get('num_original_steps')
            if error_step and total_steps:
                placement_percent = (error_step / total_steps) * 100
                step_placements.append(placement_percent)
        
        avg_placement = sum(step_placements) / len(step_placements) if step_placements else 0
        
        summary = {
            'experiment_completed': datetime.now().isoformat(),
            'total_problems': total_problems,
            'total_processed': len(results),
            'successful_injections': len(successful_results),
            'success_rate': success_rate,
            'error_type_distribution': error_types,
            'average_error_placement_percent': avg_placement,
            'placement_in_last_third': len([p for p in step_placements if p >= 67]),
            'model_used': 'gpt-4-turbo-preview',
            'error_placement_target': 'last_33_percent'
        }
        
        summary_path = f"{self.results_dir}/experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log("📈 Experiment Summary:")
        self.log(f"   Total problems: {total_problems}")
        self.log(f"   Successfully processed: {len(successful_results)}")
        self.log(f"   Success rate: {success_rate:.1%}")
        self.log(f"   Average error placement: {avg_placement:.1f}% through solution")
        self.log(f"   Errors in last third: {len([p for p in step_placements if p >= 67])}")
        self.log(f"📁 Summary saved to: {summary_path}")

def main():
    """Main execution function."""
    
    # Get the latest experiment directory
    experiments_dir = "data/experiments"
    experiment_dirs = [d for d in os.listdir(experiments_dir) if d.startswith('large_scale_')]
    
    if not experiment_dirs:
        print("❌ No experiment directories found. Run create_large_dataset.py first.")
        return
    
    # Use the most recent experiment
    latest_experiment = sorted(experiment_dirs)[-1]
    experiment_path = f"{experiments_dir}/{latest_experiment}"
    
    print(f"🎯 Using experiment: {latest_experiment}")
    
    # Load selected problems
    problems_file = f"{experiment_path}/selected_problems/complex_25plus_steps.json"
    with open(problems_file, 'r') as f:
        problems = json.load(f)
    
    print(f"📚 Loaded {len(problems)} problems for processing")
    
    # Initialize and run large-scale injection with parallel processing
    injector = LargeScaleInjector(experiment_path, max_workers=8)  # 8 parallel workers
    
    # Run with optimized settings for speed and reliability
    results = injector.run_batch_injection(
        problems=problems,
        batch_size=25,  # Larger batches for parallel processing
        checkpoint_interval=3,  # Save every 3 batches
        resume=True  # Resume from last checkpoint
    )
    
    print(f"🎉 Large-scale injection complete!")
    print(f"   Total results: {len(results)}")
    print(f"   Successful: {sum(1 for r in results if r.success)}")
    print(f"   Experiment: {latest_experiment}")

if __name__ == "__main__":
    main()