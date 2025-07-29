#!/usr/bin/env python3
"""
LateBench Batch Processing Script
Run error injection and critic evaluation on dataset batches with parallel workers.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.data_loader import LateBenchDataLoader
from core.error_injector import ErrorInjector
from core.critic import MathCritic
from data_processing.unified_schema import LateBenchExample
from utils.storage import save_examples_to_file, load_examples_from_file


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description="LateBench Batch Processing")
    
    # Dataset arguments
    parser.add_argument("--dataset", type=str, default="numinamath", 
                       help="Dataset name (default: numinamath)")
    parser.add_argument("--problem-type", type=str, default="complete",
                       help="Problem type (default: complete)")
    parser.add_argument("--input-file", type=str,
                       help="Input file path (alternative to dataset loading)")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output file path for processed examples")
    
    # Processing arguments
    parser.add_argument("--inject-errors", action="store_true",
                       help="Run error injection on examples")
    parser.add_argument("--run-critic", action="store_true",
                       help="Run critic evaluation on examples")
    parser.add_argument("--critic-mode", type=str, default="auto", 
                       choices=["auto", "original", "injected"],
                       help="Critic evaluation mode (default: auto)")
    
    # Batch processing arguments
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Number of examples to process in each batch (default: 50)")
    parser.add_argument("--injection-workers", type=int, default=4,
                       help="Number of parallel workers for error injection (default: 4)")
    parser.add_argument("--critic-workers", type=int, default=8,
                       help="Number of parallel workers for critic evaluation (default: 8)")
    parser.add_argument("--start-index", type=int, default=0,
                       help="Start processing from this example index (default: 0)")
    parser.add_argument("--max-examples", type=int,
                       help="Maximum number of examples to process")
    
    # Model arguments
    parser.add_argument("--injection-model", type=str, default="gpt-4-turbo-preview",
                       help="Model for error injection (default: gpt-4-turbo-preview)")
    parser.add_argument("--critic-model", type=str, default="gpt-4o-mini",
                       help="Model for critic evaluation (default: gpt-4o-mini)")
    
    # Options
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip examples that already have error injection/critic predictions")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output file if it exists")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be processed without actually running")
    
    return parser


def load_examples(args) -> List[LateBenchExample]:
    """Load examples from dataset or input file."""
    if args.input_file:
        print(f"📂 Loading examples from file: {args.input_file}")
        return load_examples_from_file(args.input_file)
    else:
        print(f"📂 Loading {args.dataset} dataset ({args.problem_type})...")
        loader = LateBenchDataLoader()
        return loader.load_dataset(args.dataset, args.problem_type)


def filter_examples_for_processing(examples: List[LateBenchExample], args) -> List[LateBenchExample]:
    """Filter examples based on processing requirements and skip-existing logic."""
    
    # Apply start index and max examples
    if args.start_index > 0:
        examples = examples[args.start_index:]
        print(f"📍 Starting from index {args.start_index}")
    
    if args.max_examples:
        examples = examples[:args.max_examples]
        print(f"📋 Limited to {args.max_examples} examples")
    
    # Filter based on skip-existing logic
    if args.skip_existing:
        original_count = len(examples)
        filtered_examples = []
        
        for example in examples:
            should_skip = False
            
            # Skip if error injection requested but already exists
            if args.inject_errors and example.error_injection.has_errors:
                should_skip = True
            
            # Skip if critic evaluation requested but already exists (based on mode)
            if args.run_critic and not should_skip:
                if args.critic_mode == "original" and example.critic_predictions_original:
                    should_skip = True
                elif args.critic_mode == "injected" and example.critic_predictions_injected:
                    should_skip = True
                elif args.critic_mode == "auto":
                    if example.error_injection.has_errors and example.critic_predictions_injected:
                        should_skip = True
                    elif not example.error_injection.has_errors and example.critic_predictions_original:
                        should_skip = True
            
            if not should_skip:
                filtered_examples.append(example)
        
        skipped = original_count - len(filtered_examples)
        if skipped > 0:
            print(f"⏭️  Skipped {skipped} examples that already have requested processing")
        
        examples = filtered_examples
    
    return examples


def process_batch(examples: List[LateBenchExample], batch_start: int, args) -> List[LateBenchExample]:
    """Process a batch of examples with error injection and/or critic evaluation."""
    
    batch_end = min(batch_start + args.batch_size, len(examples))
    batch = examples[batch_start:batch_end]
    batch_num = (batch_start // args.batch_size) + 1
    total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
    
    print(f"\n🔄 Processing batch {batch_num}/{total_batches} (examples {batch_start+1}-{batch_end})")
    
    # Error injection
    if args.inject_errors:
        print(f"💉 Running error injection with {args.injection_workers} workers...")
        injector = ErrorInjector(model=args.injection_model)
        batch = injector.inject_batch(batch, max_workers=args.injection_workers)
    
    # Critic evaluation  
    if args.run_critic:
        print(f"🔍 Running critic evaluation with {args.critic_workers} workers (mode: {args.critic_mode})...")
        critic = MathCritic(model=args.critic_model)
        batch = critic.evaluate_batch(batch, evaluation_mode=args.critic_mode, max_workers=args.critic_workers)
    
    # Update the main examples list
    examples[batch_start:batch_end] = batch
    
    return examples


def save_progress(examples: List[LateBenchExample], output_file: str, batch_num: int):
    """Save progress after each batch."""
    save_examples_to_file(examples, output_file)
    print(f"💾 Progress saved to {output_file} (batch {batch_num} complete)")


def print_summary(examples: List[LateBenchExample], start_time: float, args):
    """Print processing summary."""
    
    total_time = time.time() - start_time
    
    # Count results
    error_injections = sum(1 for ex in examples if ex.error_injection.has_errors)
    successful_injections = sum(1 for ex in examples if ex.error_injection.success)
    original_critics = sum(1 for ex in examples if ex.critic_predictions_original)
    injected_critics = sum(1 for ex in examples if ex.critic_predictions_injected)
    
    print(f"\n📊 BATCH PROCESSING SUMMARY")
    print(f"=" * 50)
    print(f"Total examples processed: {len(examples)}")
    print(f"Total processing time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"Average time per example: {total_time/len(examples):.2f}s")
    
    if args.inject_errors:
        print(f"\nError Injection Results:")
        print(f"  - Examples with errors: {error_injections}")
        print(f"  - Successful injections: {successful_injections}")
        print(f"  - Success rate: {successful_injections/len(examples)*100:.1f}%")
    
    if args.run_critic:
        print(f"\nCritic Evaluation Results:")
        print(f"  - Original solution evaluations: {original_critics}")
        print(f"  - Injected solution evaluations: {injected_critics}")
    
    print(f"\nOutput saved to: {args.output_file}")


def main():
    """Main batch processing function."""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.inject_errors or args.run_critic):
        print("❌ Error: Must specify at least one of --inject-errors or --run-critic")
        return 1
    
    start_time = time.time()
    
    try:
        # Load examples
        if args.resume and Path(args.output_file).exists():
            print(f"🔄 Resuming from existing output file: {args.output_file}")
            examples = load_examples_from_file(args.output_file)
        else:
            examples = load_examples(args)
        
        if not examples:
            print("❌ No examples loaded")
            return 1
        
        print(f"✅ Loaded {len(examples)} examples")
        
        # Filter examples for processing
        examples = filter_examples_for_processing(examples, args)
        
        if not examples:
            print("✅ No examples need processing (all already completed)")
            return 0
        
        print(f"🎯 {len(examples)} examples selected for processing")
        
        # Dry run check
        if args.dry_run:
            print("\n🏃 DRY RUN - Would process:")
            print(f"  - Error injection: {'Yes' if args.inject_errors else 'No'} ({args.injection_workers} workers)")
            print(f"  - Critic evaluation: {'Yes' if args.run_critic else 'No'} ({args.critic_workers} workers, {args.critic_mode} mode)")
            print(f"  - Batch size: {args.batch_size}")
            print(f"  - Total batches: {(len(examples) + args.batch_size - 1) // args.batch_size}")
            print(f"  - Output: {args.output_file}")
            return 0
        
        # Process in batches
        total_batches = (len(examples) + args.batch_size - 1) // args.batch_size
        print(f"🚀 Starting batch processing ({total_batches} batches of {args.batch_size})")
        
        for batch_start in range(0, len(examples), args.batch_size):
            examples = process_batch(examples, batch_start, args)
            batch_num = (batch_start // args.batch_size) + 1
            save_progress(examples, args.output_file, batch_num)
        
        # Final summary
        print_summary(examples, start_time, args)
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Processing interrupted by user")
        if 'examples' in locals():
            save_examples_to_file(examples, args.output_file)
            print(f"💾 Partial progress saved to {args.output_file}")
        return 1
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        if 'examples' in locals():
            save_examples_to_file(examples, args.output_file)
            print(f"💾 Partial progress saved to {args.output_file}")
        return 1


if __name__ == "__main__":
    sys.exit(main())