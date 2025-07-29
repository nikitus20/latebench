"""
Simplified parallel processing utilities for LateBench.
Core functionality: thread-safe parallel execution with rate limiting and progress tracking.
"""

import time
import threading
from typing import List, Callable, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


@dataclass
class ParallelConfig:
    """Configuration for parallel processing."""
    max_workers: int = 8
    rate_limit_per_minute: int = 100
    progress_description: str = "Processing"
    show_progress: bool = True


class ParallelProcessor:
    """Thread-safe parallel processor with rate limiting."""
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self._rate_lock = threading.Lock()
        self._last_request_times: List[float] = []
        self._results_lock = threading.Lock()
        self._progress_lock = threading.Lock()

    def process_batch(self, items: List[Any], process_func: Callable[[Any], Any]) -> List[Any]:
        """Process items in parallel with rate limiting and progress tracking."""
        
        results = []
        completed_count = 0
        successful_count = 0
        
        # Create progress bar
        if self.config.show_progress:
            pbar = tqdm(
                total=len(items), 
                desc=self.config.progress_description, 
                unit="items",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all jobs
            future_to_item = {
                executor.submit(self._process_with_rate_limit, item, process_func): item
                for item in items
            }
            
            # Process completed jobs
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    success = self._is_successful_result(result)
                    
                    # Thread-safe result collection
                    with self._results_lock:
                        results.append(result)
                    
                    # Thread-safe progress updates
                    with self._progress_lock:
                        completed_count += 1
                        if success:
                            successful_count += 1
                        
                        # Update progress bar
                        if self.config.show_progress:
                            success_rate = (successful_count / completed_count * 100) if completed_count > 0 else 0
                            pbar.set_postfix({
                                "Success": f"{successful_count}/{completed_count}",
                                "Rate": f"{success_rate:.1f}%"
                            })
                            pbar.update(1)
                    
                except Exception as e:
                    # Add error result
                    with self._results_lock:
                        results.append(self._create_error_result(item, str(e)))
                    
                    with self._progress_lock:
                        completed_count += 1
                        if self.config.show_progress:
                            pbar.update(1)
        
        if self.config.show_progress:
            pbar.close()
        
        # Sort results to maintain original order if possible
        return results

    def _process_with_rate_limit(self, item: Any, process_func: Callable[[Any], Any]) -> Any:
        """Process single item with rate limiting."""
        self._apply_rate_limit()
        return process_func(item)

    def _apply_rate_limit(self):
        """Apply rate limiting across all threads."""
        if self.config.rate_limit_per_minute <= 0:
            return
        
        with self._rate_lock:
            current_time = time.time()
            
            # Remove timestamps older than 1 minute
            self._last_request_times = [
                t for t in self._last_request_times 
                if current_time - t < 60
            ]
            
            # If we're at the rate limit, wait
            if len(self._last_request_times) >= self.config.rate_limit_per_minute:
                oldest_request = min(self._last_request_times)
                wait_time = 60 - (current_time - oldest_request) + 0.1  # Small buffer
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Record this request
            self._last_request_times.append(current_time)

    def _is_successful_result(self, result: Any) -> bool:
        """Check if result indicates success."""
        if hasattr(result, 'success'):
            return result.success
        elif isinstance(result, dict) and 'success' in result:
            return result['success']
        elif isinstance(result, dict) and 'error' in result:
            return False
        else:
            return True  # Assume success if no clear indicator

    def _create_error_result(self, item: Any, error_message: str) -> dict:
        """Create standardized error result."""
        return {
            'success': False,
            'item': item,
            'error': error_message,
            'timestamp': time.time()
        }


# Convenience functions
def parallel_map(items: List[Any], func: Callable[[Any], Any], 
                max_workers: int = 8, rate_limit: int = 100,
                description: str = "Processing") -> List[Any]:
    """Simple parallel map with rate limiting."""
    
    config = ParallelConfig(
        max_workers=max_workers,
        rate_limit_per_minute=rate_limit,
        progress_description=description
    )
    
    processor = ParallelProcessor(config)
    return processor.process_batch(items, func)


def parallel_error_injection(problems: List[dict], injector, 
                            max_workers: int = 10) -> List[Any]:
    """Parallel error injection with appropriate rate limiting."""
    
    def inject_single(problem):
        return injector.inject_error(problem)
    
    return parallel_map(
        problems, 
        inject_single, 
        max_workers=max_workers,
        rate_limit=100,  # Conservative for OpenAI API
        description="🎲 Error Injection"
    )


def parallel_critic_evaluation(problem_solution_pairs: List[tuple], critic,
                              max_workers: int = 8) -> List[Any]:
    """Parallel critic evaluation with appropriate rate limiting."""
    
    def evaluate_single(pair):
        problem, solution = pair
        return critic.evaluate_solution(problem, solution)
    
    return parallel_map(
        problem_solution_pairs,
        evaluate_single,
        max_workers=max_workers,
        rate_limit=120,  # Higher rate limit for evaluation
        description="🔍 Critic Evaluation"
    )