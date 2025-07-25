#!/usr/bin/env python3
"""
Run Batch Evaluation on All 200 MATH Level 5 Natural Error Samples

This script runs the comprehensive critic evaluation system on all 200 samples,
using parallel processing, caching, and DeltaBench metrics computation.
"""

import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.adapters.latebench_adapter import LateBenchAdapter, EvaluationPipeline
from src.metrics.deltabench import print_metrics_summary


def setup_progress_tracking():
    """Set up progress tracking for batch evaluation"""
    
    start_time = time.time()
    last_update = 0
    
    def progress_callback(progress):
        nonlocal last_update
        current_time = time.time()
        
        # Update every 5 seconds or on completion
        if current_time - last_update >= 5 or progress.completion_percentage >= 100:
            elapsed = progress.elapsed_time
            rate = progress.current_rate
            
            print(f"🔄 Progress: {progress.completion_percentage:.1f}% "
                  f"({progress.completed}/{progress.total_examples})")
            print(f"   ✅ Successful: {progress.successful} | "
                  f"❌ Failed: {progress.failed} | "
                  f"💾 Cached: {progress.cached}")
            print(f"   ⏱️  Rate: {rate:.2f} examples/sec | "
                  f"Elapsed: {elapsed:.1f}s")
            
            if progress.estimated_completion and progress.completion_percentage < 100:
                eta = progress.estimated_completion - current_time
                print(f"   🎯 ETA: {eta/60:.1f} minutes")
            
            print("-" * 60)
            last_update = current_time
    
    return progress_callback


def completion_callback(batch_summary, metrics):
    """Handle completion of batch evaluation"""
    print("\n" + "=" * 80)
    print("🎉 BATCH EVALUATION COMPLETED!")
    print("=" * 80)
    
    if batch_summary:
        print(f"📊 Batch Summary:")
        print(f"   Batch ID: {batch_summary.batch_id}")
        print(f"   Dataset: {batch_summary.dataset_name}")
        print(f"   Model: {batch_summary.model_version}")
        print(f"   Total Examples: {batch_summary.total_examples}")
        print(f"   Successful: {batch_summary.successful_evaluations}")
        print(f"   Failed: {batch_summary.failed_evaluations}")
        print(f"   Cached: {batch_summary.cached_results}")
        
        # Calculate duration
        start_time = datetime.fromisoformat(batch_summary.evaluation_start)
        end_time = datetime.fromisoformat(batch_summary.evaluation_end)
        duration = (end_time - start_time).total_seconds()
        print(f"   Duration: {duration/60:.2f} minutes")
        print(f"   Storage: {batch_summary.storage_path}")
    
    if metrics:
        print(f"\n📈 DeltaBench Metrics:")
        print_metrics_summary(metrics)
    
    print("=" * 80)


def main():
    """Main batch evaluation function"""
    
    print("🚀 Starting Batch Evaluation of 200 MATH Level 5 Natural Error Samples")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Initialize adapter
    print("🔧 Initializing LateBench Adapter...")
    adapter = LateBenchAdapter(
        storage_dir="./data/critic_store", 
        enable_dashboard_integration=True
    )
    
    # Check available datasets
    datasets = adapter.list_available_datasets()
    print(f"📁 Available datasets: {list(datasets.keys())}")
    
    # The actual dataset name according to the file
    dataset_name = 'math_level5_natural_raw'
    
    # Verify dataset exists
    if dataset_name not in datasets:
        print("❌ Could not find MATH Level 5 natural errors dataset!")
        print("Available datasets:", datasets)
        return False
    
    print(f"✅ Using dataset: {dataset_name}")
    
    # Load examples to verify count (use "errors" problem type for this dataset)
    try:
        examples = adapter.get_dataset_examples(dataset_name, "errors")
        print(f"📊 Dataset contains {len(examples)} examples")
        
        if len(examples) != 200:
            print(f"⚠️  Expected 200 examples, found {len(examples)}")
        
        # Show sample of problems
        print("\n📝 Sample problems:")
        for i, example in enumerate(examples[:3]):
            subject = example.source.metadata.get('math_split', example.source.subject)
            error_step = example.source.metadata.get('first_error_step', 'unknown')
            print(f"   {i+1}. {example.id} | Subject: {subject} | First error step: {error_step}")
        
    except Exception as e:
        print(f"❌ Error loading examples: {e}")
        return False
    
    # Check existing results
    print(f"\n💾 Checking for existing results...")
    existing_results = adapter.get_existing_results(dataset_name)
    print(f"   Found {len(existing_results)} existing evaluations")
    
    if existing_results:
        print("   Most recent evaluations:")
        for i, (example_id, stored_result) in enumerate(list(existing_results.items())[:5]):
            has_errors = "✅ Errors found" if stored_result.critic_result.has_errors else "❌ No errors"
            processing_time = stored_result.critic_result.processing_time
            print(f"     {example_id}: {has_errors} ({processing_time:.2f}s)")
    
    # Set up evaluation pipeline
    print(f"\n⚙️  Configuring evaluation pipeline...")
    pipeline = EvaluationPipeline(
        dataset_name=dataset_name,
        problem_type="errors",
        model_version="gpt-4o-mini",
        use_caching=True,
        compute_deltabench_metrics=True,
        use_first_error_cutoff=True,
        save_results=True,
        max_concurrent=15,  # Aggressive parallelization
        rate_limit=120,     # Higher rate limit
        progress_callback=setup_progress_tracking(),
        completion_callback=completion_callback
    )
    
    print(f"   Model: {pipeline.model_version}")
    print(f"   Max concurrent: {pipeline.max_concurrent}")
    print(f"   Rate limit: {pipeline.rate_limit} requests/minute")
    print(f"   Caching: {'Enabled' if pipeline.use_caching else 'Disabled'}")
    print(f"   DeltaBench metrics: {'Enabled' if pipeline.compute_deltabench_metrics else 'Disabled'}")
    print(f"   First-error cutoff: {'Enabled' if pipeline.use_first_error_cutoff else 'Disabled'}")
    
    # Estimate time if no cache
    remaining_examples = len(examples) - len(existing_results)
    if remaining_examples > 0:
        estimated_time = remaining_examples * 7 / pipeline.max_concurrent  # ~7s per example with parallelization
        print(f"   Estimated time for {remaining_examples} new evaluations: {estimated_time/60:.1f} minutes")
    
    # Confirm before starting
    print(f"\n🎯 Ready to evaluate {len(examples)} examples")
    print(f"   New evaluations needed: {remaining_examples}")
    print(f"   Using cached results: {len(existing_results)}")
    
    # Start evaluation
    print(f"\n🚀 Starting batch evaluation...")
    print("=" * 60)
    
    try:
        batch_summary = adapter.evaluate_dataset(pipeline)
        
        print(f"\n✅ Evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("LateBench Comprehensive Critic Evaluation System")
    print("=" * 80)
    
    success = main()
    
    if success:
        print("\n🎉 All 200 samples evaluated successfully!")
        print("   Results stored in ./data/critic_store/")
        print("   Check dashboard for detailed analysis")
    else:
        print("\n❌ Evaluation failed. Check logs above for details.")
    
    sys.exit(0 if success else 1)