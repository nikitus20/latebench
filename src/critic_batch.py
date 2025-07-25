"""
Batch Critic Evaluation System with Parallelization and Advanced Metrics

This module provides scalable, concurrent evaluation of mathematical reasoning
problems using LLM critics with DeltaBench-compatible metrics and intelligent
result caching.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import hashlib
from datetime import datetime

from src.critic import CriticResult, StepFormatter, LLMCritic
from src.dataset_manager import LateBenchDatasetManager
from src.data_processing.unified_schema import LateBenchExample

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BatchEvaluationConfig:
    """Configuration for batch critic evaluation"""
    model: str = "gpt-4o-mini"
    max_concurrent: int = 10
    rate_limit_per_minute: int = 100
    timeout_seconds: int = 30
    max_retries: int = 3
    save_interval: int = 50  # Save results every N evaluations
    enable_caching: bool = True
    cache_ttl_hours: int = 24


@dataclass 
class BatchProgress:
    """Progress tracking for batch evaluation"""
    total_examples: int
    completed: int
    successful: int
    failed: int
    cached: int
    start_time: float
    current_rate: float = 0.0
    estimated_completion: Optional[float] = None
    
    @property
    def completion_percentage(self) -> float:
        return (self.completed / self.total_examples * 100) if self.total_examples > 0 else 0.0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Extended result with additional metadata"""
    example_id: str
    critic_result: Optional[CriticResult]
    cached: bool
    processing_time: float
    error_message: Optional[str] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()


class AsyncCriticEvaluator:
    """Async wrapper for LLM critic evaluation"""
    
    def __init__(self, config: BatchEvaluationConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent)
        self.rate_limiter = RateLimiter(config.rate_limit_per_minute)
        
    async def evaluate_example(self, session: aiohttp.ClientSession, 
                             example: LateBenchExample) -> EvaluationResult:
        """Evaluate a single example asynchronously"""
        start_time = time.time()
        
        async with self.semaphore:  # Limit concurrent requests
            await self.rate_limiter.wait()  # Rate limiting
            
            try:
                # Convert LateBench format to critic format
                problem = example.problem.statement
                steps = [step.content for step in example.solution.steps]
                
                # Use sync critic in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor() as executor:
                    critic = LLMCritic(
                        model=self.config.model,
                        api_key=None  # Use environment variable
                    )
                    
                    critic_result = await loop.run_in_executor(
                        executor,
                        critic.evaluate_solution,
                        problem,
                        steps,
                        self.config.max_retries
                    )
                
                processing_time = time.time() - start_time
                
                return EvaluationResult(
                    example_id=example.id,
                    critic_result=critic_result,
                    cached=False,
                    processing_time=processing_time
                )
                
            except Exception as e:
                logger.error(f"Error evaluating {example.id}: {str(e)}")
                return EvaluationResult(
                    example_id=example.id,
                    critic_result=None,
                    cached=False,
                    processing_time=time.time() - start_time,
                    error_message=str(e)
                )


class RateLimiter:
    """Async rate limiter for API calls"""
    
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = requests_per_minute
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0
        self.last_request = 0.0
        
    async def wait(self):
        """Wait if necessary to respect rate limits"""
        if self.min_interval <= 0:
            return
            
        current_time = time.time()
        time_since_last = current_time - self.last_request
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request = time.time()


class CriticResultCache:
    """Intelligent caching system for critic results"""
    
    def __init__(self, cache_dir: str = "./data/cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_hours * 3600
        self.memory_cache: Dict[str, EvaluationResult] = {}
        
    def _get_cache_key(self, example: LateBenchExample, model: str) -> str:
        """Generate unique cache key for example + model combination"""
        content = f"{example.id}:{model}:{example.problem.statement}"
        content += "".join([step.content for step in example.solution.steps])
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, example: LateBenchExample, model: str) -> Optional[EvaluationResult]:
        """Get cached result if available and valid"""
        cache_key = self._get_cache_key(example, model)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            result = self.memory_cache[cache_key]
            if self._is_valid(result):
                return result
            else:
                del self.memory_cache[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                result = EvaluationResult(
                    example_id=data['example_id'],
                    critic_result=CriticResult(**data['critic_result']) if data['critic_result'] else None,
                    cached=True,
                    processing_time=data['processing_time'],
                    error_message=data.get('error_message'),
                    timestamp=data['timestamp']
                )
                
                if self._is_valid(result):
                    self.memory_cache[cache_key] = result
                    return result
                else:
                    cache_file.unlink()  # Remove expired cache
                    
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {e}")
                cache_file.unlink()  # Remove corrupted cache
        
        return None
    
    def set(self, example: LateBenchExample, model: str, result: EvaluationResult):
        """Cache evaluation result"""
        cache_key = self._get_cache_key(example, model)
        
        # Update memory cache
        self.memory_cache[cache_key] = result
        
        # Save to file cache
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data = {
                'example_id': result.example_id,
                'critic_result': result.critic_result.to_dict() if result.critic_result else None,
                'cached': result.cached,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'timestamp': result.timestamp
            }
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Error writing cache file {cache_file}: {e}")
    
    def _is_valid(self, result: EvaluationResult) -> bool:
        """Check if cached result is still valid"""
        try:
            result_time = datetime.fromisoformat(result.timestamp)
            age_seconds = (datetime.utcnow() - result_time).total_seconds()
            return age_seconds < self.ttl_seconds
        except Exception:
            return False
    
    def clear(self):
        """Clear all cached results"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class BatchCriticEvaluator:
    """Main batch evaluation orchestrator"""
    
    def __init__(self, config: Optional[BatchEvaluationConfig] = None):
        self.config = config or BatchEvaluationConfig()
        self.cache = CriticResultCache(ttl_hours=self.config.cache_ttl_hours)
        self.async_evaluator = AsyncCriticEvaluator(self.config)
        self.progress_callbacks: List[Callable[[BatchProgress], None]] = []
        
    def add_progress_callback(self, callback: Callable[[BatchProgress], None]):
        """Add callback for progress updates"""
        self.progress_callbacks.append(callback)
    
    def _notify_progress(self, progress: BatchProgress):
        """Notify all progress callbacks"""
        for callback in self.progress_callbacks:
            try:
                callback(progress)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")
    
    async def evaluate_dataset(self, examples: List[LateBenchExample],
                             save_path: Optional[str] = None) -> Dict[str, EvaluationResult]:
        """Evaluate entire dataset with parallel processing"""
        
        logger.info(f"Starting batch evaluation of {len(examples)} examples")
        
        # Initialize progress tracking
        progress = BatchProgress(
            total_examples=len(examples),
            completed=0,
            successful=0,
            failed=0,
            cached=0,
            start_time=time.time()
        )
        
        results: Dict[str, EvaluationResult] = {}
        
        # Check cache first
        if self.config.enable_caching:
            logger.info("Checking cache for existing results...")
            for example in examples:
                cached_result = self.cache.get(example, self.config.model)
                if cached_result:
                    results[example.id] = cached_result
                    progress.completed += 1
                    progress.cached += 1
                    progress.successful += 1
            
            logger.info(f"Found {progress.cached} cached results")
            self._notify_progress(progress)
        
        # Evaluate remaining examples
        remaining_examples = [ex for ex in examples if ex.id not in results]
        
        if remaining_examples:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as session:
                
                # Create evaluation tasks
                tasks = []
                for example in remaining_examples:
                    task = self.async_evaluator.evaluate_example(session, example)
                    tasks.append(task)
                
                # Process with progress tracking
                batch_size = min(self.config.max_concurrent, len(tasks))
                
                for i in range(0, len(tasks), batch_size):
                    batch_tasks = tasks[i:i + batch_size]
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result in batch_results:
                        if isinstance(result, Exception):
                            logger.error(f"Task failed: {result}")
                            progress.failed += 1
                        else:
                            results[result.example_id] = result
                            
                            if result.critic_result:
                                progress.successful += 1
                            else:
                                progress.failed += 1
                            
                            # Cache successful results
                            if self.config.enable_caching and result.critic_result:
                                example = next(ex for ex in remaining_examples if ex.id == result.example_id)
                                self.cache.set(example, self.config.model, result)
                        
                        progress.completed += 1
                        
                        # Update progress metrics
                        progress.current_rate = progress.completed / progress.elapsed_time
                        if progress.current_rate > 0:
                            remaining = progress.total_examples - progress.completed
                            progress.estimated_completion = time.time() + (remaining / progress.current_rate)
                        
                        self._notify_progress(progress)
                    
                    # Periodic save
                    if save_path and progress.completed % self.config.save_interval == 0:
                        self._save_intermediate_results(results, save_path, progress)
        
        # Final save
        if save_path:
            self._save_final_results(results, save_path, progress)
        
        logger.info(f"Batch evaluation completed: {progress.successful} successful, "
                   f"{progress.failed} failed, {progress.cached} cached")
        
        return results
    
    def _save_intermediate_results(self, results: Dict[str, EvaluationResult], 
                                 save_path: str, progress: BatchProgress):
        """Save intermediate results during batch processing"""
        try:
            intermediate_data = {
                'progress': progress.to_dict(),
                'results': {
                    result_id: {
                        'example_id': result.example_id,
                        'critic_result': result.critic_result.to_dict() if result.critic_result else None,
                        'cached': result.cached,
                        'processing_time': result.processing_time,
                        'error_message': result.error_message,
                        'timestamp': result.timestamp
                    }
                    for result_id, result in results.items()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
            
            intermediate_path = f"{save_path}.progress"
            with open(intermediate_path, 'w') as f:
                json.dump(intermediate_data, f, indent=2)
                
            logger.debug(f"Saved intermediate results to {intermediate_path}")
            
        except Exception as e:
            logger.warning(f"Error saving intermediate results: {e}")
    
    def _save_final_results(self, results: Dict[str, EvaluationResult], 
                          save_path: str, progress: BatchProgress):
        """Save final evaluation results"""
        try:
            final_data = {
                'metadata': {
                    'total_examples': progress.total_examples,
                    'successful': progress.successful,
                    'failed': progress.failed,
                    'cached': progress.cached,
                    'processing_time': progress.elapsed_time,
                    'model_used': self.config.model,
                    'timestamp': datetime.utcnow().isoformat()
                },
                'results': {
                    result_id: {
                        'example_id': result.example_id,
                        'critic_result': result.critic_result.to_dict() if result.critic_result else None,
                        'cached': result.cached,
                        'processing_time': result.processing_time,
                        'error_message': result.error_message,
                        'timestamp': result.timestamp
                    }
                    for result_id, result in results.items()
                }
            }
            
            with open(save_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            
            logger.info(f"Saved final results to {save_path}")
            
            # Clean up intermediate file
            intermediate_path = f"{save_path}.progress"
            if Path(intermediate_path).exists():
                Path(intermediate_path).unlink()
                
        except Exception as e:
            logger.error(f"Error saving final results: {e}")


# Convenience functions
async def evaluate_dataset_batch(dataset_name: str, 
                               config: Optional[BatchEvaluationConfig] = None,
                               save_results: bool = True) -> Dict[str, EvaluationResult]:
    """Evaluate an entire LateBench dataset with batch processing"""
    
    # Load dataset
    manager = LateBenchDatasetManager()
    if not manager.load_dataset(dataset_name):
        raise ValueError(f"Failed to load dataset: {dataset_name}")
    
    examples = manager.get_current_examples()
    if not examples:
        raise ValueError(f"No examples found in dataset: {dataset_name}")
    
    # Create evaluator
    evaluator = BatchCriticEvaluator(config)
    
    # Set up save path
    save_path = None
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"./data/critic_results/{dataset_name}_batch_{timestamp}.json"
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    results = await evaluator.evaluate_dataset(examples, save_path)
    
    return results


def run_batch_evaluation(dataset_name: str, 
                        config: Optional[BatchEvaluationConfig] = None) -> Dict[str, EvaluationResult]:
    """Synchronous wrapper for batch evaluation"""
    return asyncio.run(evaluate_dataset_batch(dataset_name, config))


if __name__ == "__main__":
    # Example usage
    config = BatchEvaluationConfig(
        model="gpt-4o-mini",
        max_concurrent=5,
        rate_limit_per_minute=50,
        enable_caching=True
    )
    
    # Run batch evaluation
    results = run_batch_evaluation("math_level5_natural_errors", config)
    print(f"Evaluated {len(results)} examples")