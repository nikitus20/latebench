#!/usr/bin/env python3
"""
LateBench Critic Evaluation with Step-Level DeltaBench Metrics

This script evaluates the LateBench critic system on batch error injection results
using proper step-level DeltaBench metrics with first-error cutoff logic.

Key Features:
- Parallel critic evaluation with progress tracking
- Step-level precision/recall calculation (not problem-level)
- First-error cutoff logic to avoid evaluation bias
- DeltaBench-compatible metrics output
- Data leakage prevention (no ground truth in critic input)

Usage:
    python run_critic_on_batch_data.py

Results are saved to: ./data/critic_evaluation_results_<timestamp>/
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.critic import LLMCritic, evaluate_single_example
from src.metrics.deltabench import DeltaBenchEvaluator, print_metrics_summary
from src.data_processing.unified_schema import LateBenchExample


class ParallelCriticEvaluator:
    """Parallel critic evaluation for batch injection results."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._thread_local = threading.local()
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()
    
    def _get_critic(self):
        """Get thread-local critic instance."""
        if not hasattr(self._thread_local, 'critic'):
            self._thread_local.critic = LLMCritic()
        return self._thread_local.critic
    
    def _process_single_example(self, problem_data):
        """Process a single problem with critic evaluation."""
        try:
            raw_example, example_id = problem_data
            
            # Run critic evaluation
            start_time = time.time()
            critic_result = evaluate_single_example(raw_example)
            processing_time = time.time() - start_time
            
            return {
                'example_id': example_id,
                'critic_result': critic_result.to_dict() if critic_result else None,
                'processing_time': processing_time,
                'success': True,  # If we got a result, it's successful
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'example_id': example_id,
                'critic_result': None,
                'processing_time': 0,
                'success': False,
                'error_message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_batch(self, problems: List[tuple]) -> List[Dict[str, Any]]:
        """Evaluate a batch of problems using parallel workers."""
        results = []
        completed_count = 0
        successful_count = 0
        
        # Create progress bar
        pbar = tqdm(total=len(problems), desc="🔍 Critic Evaluation", unit="problems",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_problem = {
                executor.submit(self._process_single_example, problem): problem
                for problem in problems
            }
            
            # Process completed jobs
            for future in as_completed(future_to_problem):
                problem = future_to_problem[future]
                _, example_id = problem
                
                try:
                    result = future.result()
                    
                    # Thread-safe result collection
                    with self._results_lock:
                        results.append(result)
                    
                    # Thread-safe progress updates
                    with self._progress_lock:
                        completed_count += 1
                        if result['success']:
                            successful_count += 1
                        
                        # Update progress bar
                        success_rate = (successful_count / completed_count * 100) if completed_count > 0 else 0
                        pbar.set_postfix({
                            "Success": f"{successful_count}/{completed_count}",
                            "Rate": f"{success_rate:.1f}%",
                            "Workers": self.max_workers
                        })
                        pbar.update(1)
                    
                except Exception as e:
                    with self._progress_lock:
                        completed_count += 1
                        pbar.update(1)
                    print(f"Error processing {example_id}: {e}")
        
        pbar.close()
        
        # Sort results by original order
        results.sort(key=lambda x: x['example_id'])
        return results


def convert_injection_result_to_critic_format(injection_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert batch injection result to format expected by critic."""
    
    if not injection_data.get('success', False):
        return None
    
    original_example = injection_data['original_example']
    injection_result = injection_data.get('injection_result', {})
    
    if not injection_result:
        return None
    
    # Extract original problem data
    original_steps = original_example.get('solution', {}).get('steps', [])
    
    # Extract modified solution from injection result
    modified_solution = injection_result.get('modified_solution', {})
    modified_steps = modified_solution.get('steps', [])
    
    # Create the format expected by critic - ONLY provide problem statement and modified solution content
    # DO NOT provide original solution or error markers to avoid data leakage
    raw_example = {
        'original_problem': {
            'problem': original_example.get('problem', {}).get('statement', '')
            # REMOVED: parsed_steps - this was giving the critic the correct solution!
        },
        'modified_solution': {
            'steps': [
                {
                    'step_num': step.get('step_num', i+1),
                    'content': step.get('content', '')
                    # REMOVED: modified and error flags - this was telling the critic the answers!
                }
                for i, step in enumerate(modified_steps)
            ]
        }
    }
    
    return raw_example


def convert_injection_to_latebench_format(injection_data: Dict[str, Any]) -> Optional[LateBenchExample]:
    """Convert batch injection result to LateBenchExample format with proper error annotations."""
    
    if not injection_data.get('success', False):
        return None
    
    original_example = injection_data['original_example']
    injection_result = injection_data.get('injection_result', {})
    
    if not injection_result:
        return None
    
    # Get modified solution with error markers
    modified_solution = injection_result.get('modified_solution', {})
    modified_steps = modified_solution.get('steps', [])
    
    # Create LateBenchExample-compatible steps with proper error marking
    from src.data_processing.unified_schema import LateBenchStep, LateBenchSolution, LateBenchProblem
    
    steps = []
    for step_data in modified_steps:
        step = LateBenchStep(
            step_number=step_data.get('step_num', 0),
            content=step_data.get('content', ''),
            importance='medium',
            reasoning_type='unknown',
            is_modified=step_data.get('modified', False),
            is_error=step_data.get('error', False)  # This is the key ground truth
        )
        steps.append(step)
    
    # Create solution object
    solution = LateBenchSolution(
        steps=steps,
        final_answer=modified_solution.get('final_answer', ''),
        total_steps=len(steps),
        solution_method='analytical'
    )
    
    # Create problem object
    problem_data = original_example.get('problem', {})
    problem = LateBenchProblem(statement=problem_data.get('statement', ''))
    
    # Create LateBenchExample
    from src.data_processing.unified_schema import LateBenchSource, LateBenchErrorInjection, LateBenchProcessing
    
    # Create source metadata
    source = LateBenchSource(
        dataset=original_example.get('source', {}).get('dataset', ''),
        original_id=original_example.get('source', {}).get('original_id', ''),
        difficulty=original_example.get('source', {}).get('difficulty', 0.0),
        subject=original_example.get('source', {}).get('subject', ''),
        competition=original_example.get('source', {}).get('competition', ''),
        year=original_example.get('source', {}).get('year'),
        metadata=original_example.get('source', {}).get('metadata', {})
    )
    
    # Create error injection info
    error_injection = LateBenchErrorInjection(
        has_errors=True,  # We injected an error
        error_info=injection_result.get('error_analysis'),
        manual_attempts=[],
        final_decision=None,
        decision_timestamp=None,
        custom_suggestions=[]
    )
    
    # Create processing info
    processing = LateBenchProcessing(
        added_to_latebench=injection_data.get('timestamp', ''),
        last_modified=injection_data.get('timestamp', ''),
        status='processed',
        processor_version='1.0'
    )
    
    return LateBenchExample(
        id=injection_data['example_id'],
        source=source,
        problem=problem,
        solution=solution,
        error_injection=error_injection,
        processing=processing
    )


def calculate_proper_deltabench_metrics(injection_results: List[Dict[str, Any]], 
                                      critic_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate proper step-level DeltaBench metrics with first-error cutoff."""
    
    # Convert injection results to LateBench format
    examples = []
    critic_results_dict = {}
    
    print("🔄 Converting injection results to LateBench format...")
    
    # Create mapping from critic results
    critic_lookup = {result['example_id']: result for result in critic_results if result.get('success', False)}
    
    for injection_data in injection_results:
        if not injection_data.get('success', False):
            continue
            
        example_id = injection_data['example_id']
        if example_id not in critic_lookup:
            continue
            
        # Convert to LateBench format
        latebench_example = convert_injection_to_latebench_format(injection_data)
        if not latebench_example:
            continue
            
        examples.append(latebench_example)
        
        # Convert critic result to CriticResult object
        critic_data = critic_lookup[example_id]['critic_result']
        if not critic_data:
            continue
            
        from src.critic import CriticResult
        critic_result = CriticResult(
            has_errors=critic_data.get('has_errors', False),
            error_steps=critic_data.get('error_steps', []),
            explanations=critic_data.get('explanations', {}),
            raw_response=critic_data.get('raw_response', ''),
            model_used=critic_data.get('model_used', ''),
            processing_time=critic_data.get('processing_time', 0.0)
        )
        
        critic_results_dict[example_id] = critic_result
    
    print(f"✅ Converted {len(examples)} examples for DeltaBench evaluation")
    
    # Use DeltaBench evaluator with first-error cutoff
    evaluator = DeltaBenchEvaluator(use_first_error_cutoff=True)
    metrics = evaluator.evaluate_batch(examples, critic_results_dict)
    
    return metrics




def main():
    """Main function to run critic evaluation on batch injection data."""
    
    print("🔍 LateBench Critic Evaluation on Batch Injection Results")
    print("=" * 70)
    
    # Load batch injection results
    batch_results_path = "./data/batch_injection_results_20250727_233901/general_injection_results.json"
    
    if not os.path.exists(batch_results_path):
        print(f"❌ Batch results file not found: {batch_results_path}")
        print("Please run the batch injection first.")
        return
    
    print(f"📂 Loading batch injection results from: {batch_results_path}")
    
    with open(batch_results_path, 'r') as f:
        injection_results = json.load(f)
    
    print(f"📊 Loaded {len(injection_results)} injection results")
    
    # Convert to critic format and filter successful injections
    problems_for_critic = []
    successful_injections = 0
    
    for injection_data in injection_results:
        if injection_data.get('success', False):
            raw_example = convert_injection_result_to_critic_format(injection_data)
            if raw_example:
                problems_for_critic.append((raw_example, injection_data['example_id']))
                successful_injections += 1
    
    print(f"✅ {successful_injections} successful injections ready for critic evaluation")
    print(f"🎯 Running critic evaluation on {len(problems_for_critic)} problems...")
    print("=" * 70)
    
    # Run parallel critic evaluation
    critic_evaluator = ParallelCriticEvaluator(max_workers=8)
    
    start_time = time.time()
    critic_results = critic_evaluator.evaluate_batch(problems_for_critic)
    total_time = time.time() - start_time
    
    print(f"\n🎉 Critic Evaluation Complete!")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"   📊 {len(critic_results)} problems evaluated")
    
    # Calculate proper step-level DeltaBench metrics
    print(f"\n📈 Calculating proper step-level DeltaBench metrics...")
    metrics = calculate_proper_deltabench_metrics(injection_results, critic_results)
    
    # Display results using proper DeltaBench format
    print("\n📊 DeltaBench Evaluation Results:")
    print_metrics_summary(metrics)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./data/critic_evaluation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save critic results
    with open(f"{output_dir}/critic_results.json", 'w') as f:
        json.dump(critic_results, f, indent=2, default=str)
    
    # Save metrics
    with open(f"{output_dir}/deltabench_metrics.json", 'w') as f:
        json.dump(metrics.to_dict(), f, indent=2, default=str)
    
    print(f"💾 Results saved to: {output_dir}")
    print(f"📊 Metrics summary: {output_dir}/deltabench_metrics.json")
    print(f"🔍 Detailed results: {output_dir}/critic_results.json")


if __name__ == "__main__":
    main()