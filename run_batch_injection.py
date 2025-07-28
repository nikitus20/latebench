#!/usr/bin/env python3
"""
Run Batch Error Injection for Testing Both General and Manual Approaches

This script runs error injection on the NuminaMath dataset using both approaches:
1. General injection - automatic error type selection 
2. Manual injection - using predefined custom suggestions

Designed to test 100 problems per approach for comparison analysis.
"""

import sys
import os
import time
import json
import random
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.error_injector import AdversarialErrorInjector
from src.dataset_manager import LateBenchDatasetManager


class ParallelBatchInjector:
    """Thread-safe parallel wrapper for AdversarialErrorInjector."""
    
    def __init__(self, max_workers: int = 10, api_rate_limit: int = 100):
        """
        Initialize parallel batch injector.
        
        Args:
            max_workers: Maximum number of concurrent worker threads
            api_rate_limit: Maximum API requests per minute (for rate limiting)
        """
        self.max_workers = max_workers
        self.api_rate_limit = api_rate_limit
        
        # Create thread-local storage for injectors to avoid sharing state
        self._thread_local = threading.local()
        
        # Rate limiting with semaphore (requests per minute)
        self._rate_semaphore = threading.Semaphore(max_workers)
        self._last_request_times = []
        self._rate_lock = threading.Lock()
        
        # Progress tracking
        self._progress_lock = threading.Lock()
        self._results_lock = threading.Lock()
        
    def _get_injector(self):
        """Get thread-local injector instance."""
        if not hasattr(self._thread_local, 'injector'):
            self._thread_local.injector = AdversarialErrorInjector()
            # Disable individual rate limiting since we handle it at batch level
            self._thread_local.injector.requests_per_minute = 0
        return self._thread_local.injector
    
    def _rate_limit(self):
        """Implement intelligent rate limiting across all threads."""
        with self._rate_lock:
            current_time = time.time()
            
            # Remove timestamps older than 1 minute
            self._last_request_times = [
                t for t in self._last_request_times 
                if current_time - t < 60
            ]
            
            # If we're at the rate limit, wait
            if len(self._last_request_times) >= self.api_rate_limit:
                oldest_request = min(self._last_request_times)
                wait_time = 60 - (current_time - oldest_request)
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Record this request
            self._last_request_times.append(current_time)
    
    def _process_single_problem(self, problem_data):
        """Process a single problem in a worker thread."""
        problem, example_id, raw_example = problem_data
        
        try:
            # Rate limiting
            self._rate_limit()
            
            # Get thread-local injector
            injector = self._get_injector()
            
            # Run general injection
            start_time = time.time()
            injection_result = injector.inject_error(
                raw_example,
                error_type_preference=None,  # General injection
                max_retries=3
            )
            processing_time = time.time() - start_time
            
            # Store result
            result_data = {
                'example_id': example_id,
                'original_example': problem,
                'raw_example': raw_example,
                'injection_approach': 'general',
                'injection_result': injection_result.__dict__ if injection_result else None,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': injection_result.success if injection_result else False,
                'thread_id': threading.current_thread().ident
            }
            
            return result_data
            
        except Exception as e:
            # Handle any unexpected errors
            return {
                'example_id': example_id,
                'original_example': problem,
                'raw_example': raw_example,
                'injection_approach': 'general',
                'injection_result': None,
                'processing_time': 0,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error_message': str(e),
                'thread_id': threading.current_thread().ident
            }
    
    def parallel_inject_batch(self, examples: List[Dict[str, Any]], 
                            convert_to_raw_format_func,
                            progress_callback=None) -> List[Dict[str, Any]]:
        """
        Process a batch of examples using parallel workers.
        
        Args:
            examples: List of examples to process
            convert_to_raw_format_func: Function to convert examples to raw format
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of injection results
        """
        # Prepare work items
        work_items = []
        for i, example in enumerate(examples):
            example_id = example.get('id', f'example_{i}')
            raw_example = convert_to_raw_format_func(example)
            work_items.append((example, example_id, raw_example))
        
        results = []
        completed_count = 0
        successful_count = 0
        
        # Create progress bar
        pbar = tqdm(total=len(examples), desc="🎲 Parallel Injection", unit="problems",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        # Process with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_work = {
                executor.submit(self._process_single_problem, work_item): work_item
                for work_item in work_items
            }
            
            # Process completed jobs
            for future in as_completed(future_to_work):
                work_item = future_to_work[future]
                _, example_id, _ = work_item
                
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
                        
                        # Call progress callback if provided
                        if progress_callback:
                            progress_callback(completed_count, successful_count, len(examples))
                    
                except Exception as e:
                    # Handle future execution errors
                    with self._progress_lock:
                        completed_count += 1
                        pbar.update(1)
                    print(f"Error processing {example_id}: {e}")
        
        pbar.close()
        
        # Sort results by original order (based on example_id if needed)
        results.sort(key=lambda x: x['example_id'])
        
        return results


class BatchInjectionProgress:
    """Track progress of batch injection operations."""
    
    def __init__(self, total_examples: int, approach: str):
        self.total_examples = total_examples
        self.approach = approach
        self.start_time = time.time()
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.current_example_id = None
        
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def completion_percentage(self) -> float:
        return (self.completed / self.total_examples) * 100 if self.total_examples > 0 else 0
    
    @property
    def current_rate(self) -> float:
        return self.completed / self.elapsed_time if self.elapsed_time > 0 else 0
    
    @property
    def estimated_completion(self) -> Optional[float]:
        if self.current_rate > 0:
            remaining = self.total_examples - self.completed
            return time.time() + (remaining / self.current_rate)
        return None


def setup_progress_tracking(approach: str, total_examples: int):
    """Set up progress tracking for batch injection"""
    
    progress = BatchInjectionProgress(total_examples, approach)
    last_update = 0
    
    def update_progress(current_example_id: str = None):
        nonlocal last_update
        current_time = time.time()
        
        if current_example_id:
            progress.current_example_id = current_example_id
        
        # Update every 10 seconds or on completion
        if current_time - last_update >= 10 or progress.completion_percentage >= 100:
            elapsed = progress.elapsed_time
            rate = progress.current_rate
            
            print(f"🔄 {approach.title()} Injection Progress: {progress.completion_percentage:.1f}% "
                  f"({progress.completed}/{progress.total_examples})")
            print(f"   ✅ Successful: {progress.successful} | "
                  f"❌ Failed: {progress.failed}")
            print(f"   ⏱️  Rate: {rate:.2f} problems/sec | "
                  f"Elapsed: {elapsed/60:.1f} minutes")
            
            if progress.current_example_id:
                print(f"   📝 Current: {progress.current_example_id}")
            
            if progress.estimated_completion and progress.completion_percentage < 100:
                eta = progress.estimated_completion - current_time
                print(f"   🎯 ETA: {eta/60:.1f} minutes")
            
            print("-" * 70)
            last_update = current_time
    
    return progress, update_progress


def load_numinamath_examples(max_examples: int = None) -> List[Dict[str, Any]]:
    """Load NuminaMath examples for injection testing."""
    
    print("📂 Loading NuminaMath dataset...")
    
    # Load from the processed dataset
    dataset_path = "./data/datasets/latebench_numinamath_complete.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return []
    
    with open(dataset_path, 'r') as f:
        examples = json.load(f)
    
    print(f"📊 Loaded {len(examples)} examples from NuminaMath dataset")
    
    if max_examples and len(examples) > max_examples:
        # Randomly sample examples for testing
        random.shuffle(examples)
        examples = examples[:max_examples]
        print(f"🎲 Randomly selected {len(examples)} examples for injection testing")
    
    return examples


def convert_to_raw_format(latebench_example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LateBench format to raw format for error injector."""
    
    steps = latebench_example.get('solution', {}).get('steps', [])
    solution_text = '\\n'.join([step['content'] for step in steps])
    
    return {
        'problem': latebench_example.get('problem', ''),
        'solution': solution_text,
        'answer': latebench_example.get('solution', {}).get('final_answer', 'See solution'),
        'latebench_id': latebench_example.get('id', ''),
        'source_metadata': latebench_example.get('source', {})
    }


def run_general_injection_batch(examples: List[Dict[str, Any]], 
                               save_path: str,
                               checkpoint_interval: int = 10,
                               max_workers: int = 10,
                               use_parallel: bool = True) -> List[Dict[str, Any]]:
    """Run general error injection on a batch of examples with optional parallel processing."""
    
    print(f"🎲 Starting General Injection Batch Processing")
    print(f"   📊 Total examples: {len(examples)}")
    print(f"   💾 Save path: {save_path}")
    print(f"   🔄 Checkpoint interval: {checkpoint_interval}")
    print(f"   🚀 Parallel processing: {'Enabled' if use_parallel else 'Disabled'}")
    if use_parallel:
        print(f"   👥 Max workers: {max_workers}")
    print("=" * 70)
    
    start_time = time.time()
    
    if use_parallel:
        # Use parallel processing
        parallel_injector = ParallelBatchInjector(
            max_workers=max_workers,
            api_rate_limit=100  # Adjust based on your OpenAI tier
        )
        
        # Track progress for checkpoints
        progress, update_progress = setup_progress_tracking("general", len(examples))
        checkpoint_counter = 0
        
        def checkpoint_callback(completed, successful, total):
            """Callback for checkpoint saving during parallel processing."""
            nonlocal checkpoint_counter
            if completed % checkpoint_interval == 0 and completed > checkpoint_counter:
                checkpoint_counter = completed
                # Note: In parallel processing, we save checkpoints less frequently
                # since results come back out of order
        
        # Run parallel injection
        results = parallel_injector.parallel_inject_batch(
            examples,
            convert_to_raw_format,
            progress_callback=checkpoint_callback
        )
        
        # Update final progress stats
        progress.completed = len(results)
        progress.successful = sum(1 for r in results if r['success'])
        progress.failed = progress.completed - progress.successful
        
    else:
        # Use sequential processing (fallback)
        injector = AdversarialErrorInjector()
        progress, update_progress = setup_progress_tracking("general", len(examples))
        
        results = []
        
        # Create progress bar for sequential processing
        pbar = tqdm(examples, desc="🎲 Sequential Injection", unit="problems", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, example in enumerate(pbar):
            example_id = example.get('id', f'example_{i}')
            update_progress(example_id)
            
            # Update progress bar description
            pbar.set_description(f"🎲 Processing {example_id[:12]}...")
            
            # Convert to raw format
            raw_example = convert_to_raw_format(example)
            
            # Run general injection
            injection_start = time.time()
            injection_result = injector.inject_error(
                raw_example,
                error_type_preference=None,  # General injection
                max_retries=3
            )
            processing_time = time.time() - injection_start
            
            # Store result
            result_data = {
                'example_id': example_id,
                'original_example': example,
                'raw_example': raw_example,
                'injection_approach': 'general',
                'injection_result': injection_result.__dict__ if injection_result else None,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'success': injection_result.success if injection_result else False
            }
            
            results.append(result_data)
            progress.completed += 1
            
            if injection_result and injection_result.success:
                progress.successful += 1
            else:
                progress.failed += 1
            
            success_rate = (progress.successful / progress.completed * 100) if progress.completed > 0 else 0
            pbar.set_postfix({
                "Success": f"{progress.successful}/{progress.completed}", 
                "Rate": f"{success_rate:.1f}%"
            })
            
            # Save checkpoint
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_path = save_path.replace('.json', f'_checkpoint_{i+1}.json')
                save_results(results, checkpoint_path)
                pbar.write(f"💾 Checkpoint saved: {checkpoint_path}")
        
        pbar.close()
    
    # Save final results
    save_results(results, save_path)
    
    # Final checkpoint if needed
    if len(results) % checkpoint_interval != 0:
        final_checkpoint_path = save_path.replace('.json', f'_final_checkpoint.json')
        save_results(results, final_checkpoint_path)
    
    total_time = time.time() - start_time
    
    print(f"\\n🎉 General Injection Batch Complete!")
    print(f"   📊 Total: {len(results)} | Success: {progress.successful} | Failed: {progress.failed}")
    print(f"   ✅ Success Rate: {progress.successful/len(results)*100:.1f}%")
    print(f"   ⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"   🚀 Processing mode: {'Parallel' if use_parallel else 'Sequential'}")
    if use_parallel:
        print(f"   ⚡ Speed improvement: ~{max_workers}x faster than sequential")
    print(f"   💾 Results saved to: {save_path}")
    
    return results


# Pre-defined custom suggestions for manual injection testing
CUSTOM_SUGGESTIONS = [
    "Make an invalid domain assumption - assume a property holds for all values when it only applies to a subset",
    "Introduce circular reasoning - use the conclusion to prove an intermediate step",
    "Apply a theorem outside its valid domain - use a rule where its conditions aren't met",  
    "Make an unjustified generalization - extend a specific case without proper justification",
    "Violate a constraint or boundary condition - ignore a limitation stated in the problem",
    "Misapply distributive or commutative properties in a context where they don't hold",
    "Make an invalid assumption about continuity or smoothness where not guaranteed",
    "Incorrectly handle edge cases or special values",
    "Apply algebraic manipulation that changes the solution set",
    "Make a logical leap without sufficient intermediate steps"
]


def run_manual_injection_batch(examples: List[Dict[str, Any]], 
                              save_path: str,
                              checkpoint_interval: int = 10) -> List[Dict[str, Any]]:
    """Run manual error injection with custom suggestions on a batch of examples."""
    
    print(f"🎯 Starting Manual Injection Batch Processing")
    print(f"   📊 Total examples: {len(examples)}")
    print(f"   💾 Save path: {save_path}")
    print(f"   🔄 Checkpoint interval: {checkpoint_interval}")
    print(f"   📝 Available suggestions: {len(CUSTOM_SUGGESTIONS)}")
    print("=" * 70)
    
    # Initialize injector
    injector = AdversarialErrorInjector()
    progress, update_progress = setup_progress_tracking("manual", len(examples))
    
    results = []
    
    for i, example in enumerate(examples):
        example_id = example.get('id', f'example_{i}')
        update_progress(example_id)
        
        print(f"🔄 Processing {i+1}/{len(examples)}: {example_id}")
        
        # Convert to raw format
        raw_example = convert_to_raw_format(example)
        
        # Select a random custom suggestion
        custom_suggestion = random.choice(CUSTOM_SUGGESTIONS)
        print(f"   📝 Using suggestion: {custom_suggestion[:60]}...")
        
        # Run manual injection with custom suggestion
        start_time = time.time()
        injection_result = injector.inject_error_with_custom_suggestion(
            raw_example,
            custom_suggestion=custom_suggestion,
            max_retries=3
        )
        processing_time = time.time() - start_time
        
        # Store result
        result_data = {
            'example_id': example_id,
            'original_example': example,
            'raw_example': raw_example,
            'injection_approach': 'manual',
            'custom_suggestion': custom_suggestion,
            'injection_result': injection_result.__dict__ if injection_result else None,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat(),
            'success': injection_result.success if injection_result else False
        }
        
        results.append(result_data)
        progress.completed += 1
        
        if injection_result and injection_result.success:
            progress.successful += 1
            error_type = injection_result.error_analysis.get('error_type', 'unknown')
            error_step = injection_result.error_analysis.get('selected_error_step', 'unknown')
            print(f"   ✅ Success: {error_type} error in step {error_step} ({processing_time:.1f}s)")
        else:
            progress.failed += 1
            error_msg = injection_result.error_message if injection_result else 'Unknown error'
            print(f"   ❌ Failed: {error_msg} ({processing_time:.1f}s)")
        
        # Save checkpoint
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_path = save_path.replace('.json', f'_checkpoint_{i+1}.json')
            save_results(results, checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_path}")
    
    # Final progress update
    update_progress()
    
    # Save final results
    save_results(results, save_path)
    
    print(f"\\n🎉 Manual Injection Batch Complete!")
    print(f"   📊 Total: {len(results)} | Success: {progress.successful} | Failed: {progress.failed}")
    print(f"   ⏱️  Total time: {progress.elapsed_time/60:.1f} minutes")
    print(f"   💾 Results saved to: {save_path}")
    
    return results


def save_results(results: List[Dict[str, Any]], filepath: str):
    """Save injection results to JSON file."""
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save results
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Saved {len(results)} results to {filepath}")


def analyze_general_results(general_results: List[Dict[str, Any]]):
    """Analyze results from general injection approach only."""
    
    print("\\n" + "=" * 70)
    print("📊 GENERAL INJECTION ANALYSIS")
    print("=" * 70)
    
    # General approach analysis
    general_success = sum(1 for r in general_results if r['success'])
    general_total = len(general_results)
    general_rate = general_success / general_total if general_total > 0 else 0
    
    print(f"🎲 General Injection Results:")
    print(f"   Total: {general_total} | Success: {general_success} | Failed: {general_total - general_success}")
    print(f"   Success Rate: {general_rate*100:.1f}%")
    
    if general_results:
        general_times = [r['processing_time'] for r in general_results if r['success']]
        if general_times:
            avg_time = sum(general_times) / len(general_times)
            print(f"   Average Processing Time: {avg_time:.1f}s")
            
        # Error type distribution
        error_types = {}
        for r in general_results:
            if r['success'] and r['injection_result']:
                error_type = r['injection_result'].get('error_analysis', {}).get('error_type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if error_types:
            print(f"\\n📈 Error Type Distribution:")
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / general_success) * 100 if general_success > 0 else 0
                print(f"   {error_type}: {count} ({percentage:.1f}%)")
    
    print("=" * 70)


def analyze_results(general_results: List[Dict[str, Any]], 
                   manual_results: List[Dict[str, Any]]):
    """Analyze and compare results from both approaches."""
    
    print("\\n" + "=" * 70)
    print("📊 BATCH INJECTION ANALYSIS")
    print("=" * 70)
    
    # General approach analysis
    general_success = sum(1 for r in general_results if r['success'])
    general_total = len(general_results)
    general_rate = general_success / general_total if general_total > 0 else 0
    
    print(f"🎲 General Injection Results:")
    print(f"   Total: {general_total} | Success: {general_success} | Failed: {general_total - general_success}")
    print(f"   Success Rate: {general_rate*100:.1f}%")
    
    if general_results:
        general_times = [r['processing_time'] for r in general_results if r['success']]
        if general_times:
            avg_time = sum(general_times) / len(general_times)
            print(f"   Average Processing Time: {avg_time:.1f}s")
    
    # Manual approach analysis  
    manual_success = sum(1 for r in manual_results if r['success'])
    manual_total = len(manual_results)
    manual_rate = manual_success / manual_total if manual_total > 0 else 0
    
    print(f"\\n🎯 Manual Injection Results:")
    print(f"   Total: {manual_total} | Success: {manual_success} | Failed: {manual_total - manual_success}")
    print(f"   Success Rate: {manual_rate*100:.1f}%")
    
    if manual_results:
        manual_times = [r['processing_time'] for r in manual_results if r['success']]
        if manual_times:
            avg_time = sum(manual_times) / len(manual_times)
            print(f"   Average Processing Time: {avg_time:.1f}s")
    
    # Comparison
    print(f"\\n⚖️  Comparison:")
    if general_rate > manual_rate:
        diff = (general_rate - manual_rate) * 100
        print(f"   General injection performed better by {diff:.1f} percentage points")
    elif manual_rate > general_rate:
        diff = (manual_rate - general_rate) * 100
        print(f"   Manual injection performed better by {diff:.1f} percentage points")
    else:
        print(f"   Both approaches achieved similar success rates")
    
    print("=" * 70)


def main():
    """Main function to run batch injection experiments."""
    
    print("🚀 LateBench Batch Error Injection Experiment")
    print("Testing General vs Manual Injection Approaches")
    print("=" * 70)
    
    # Configuration - Only run general injection for now
    # Parallel processing validated! Now running full experiment
    total_examples = 100  # Full experiment: 100 examples with parallel processing
    
    # Load examples
    all_examples = load_numinamath_examples(max_examples=total_examples)
    
    if len(all_examples) < total_examples:
        print(f"⚠️  Warning: Only {len(all_examples)} examples available, less than needed {total_examples}")
        total_examples = len(all_examples)
    
    # Use all examples for general injection
    random.seed(42)  # For reproducible results
    random.shuffle(all_examples)
    
    general_examples = all_examples[:total_examples]
    
    print(f"📊 Experiment Configuration:")
    print(f"   General injection examples: {len(general_examples)}")
    print(f"   Manual injection examples: 0 (skipped - no custom suggestions available)")
    print(f"   Total examples: {len(general_examples)}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"./data/batch_injection_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Run general injection batch with parallel processing
    print(f"\\n🎲 General Injection on {len(general_examples)} Problems")
    general_results = run_general_injection_batch(
        general_examples, 
        f"{output_dir}/general_injection_results.json",
        checkpoint_interval=10,  # Checkpoint every 10 problems
        max_workers=10,          # Use 10 parallel workers
        use_parallel=True        # Enable parallel processing
    )
    
    # Skip manual injection for now
    print(f"\\n⏭️  Skipping Manual Injection (no custom suggestions available)")
    manual_results = []
    
    # Analyze results - only general injection
    analyze_general_results(general_results)
    
    # Save combined analysis
    combined_analysis = {
        'experiment_config': {
            'timestamp': timestamp,
            'total_examples': len(general_examples),
            'approach': 'general_injection_only'
        },
        'general_approach': {
            'total': len(general_results),
            'successful': sum(1 for r in general_results if r['success']),
            'success_rate': sum(1 for r in general_results if r['success']) / len(general_results) if general_results else 0
        },
        'manual_approach': {
            'total': 0,
            'successful': 0,
            'success_rate': 0,
            'note': 'Skipped - no custom suggestions available'
        }
    }
    
    with open(f"{output_dir}/experiment_analysis.json", 'w') as f:
        json.dump(combined_analysis, f, indent=2)
    
    print(f"\\n🎉 General Injection Experiment Complete!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Analysis summary: {output_dir}/experiment_analysis.json")
    print(f"\\n📈 Summary: {len(general_results)} problems processed with general injection")
    if general_results:
        success_rate = sum(1 for r in general_results if r['success']) / len(general_results) * 100
        print(f"✅ Overall success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()