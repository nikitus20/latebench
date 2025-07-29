#!/usr/bin/env python3
"""
Simplified error injection script for LateBench.
Run error injection on mathematical problems with optional custom suggestions.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.error_injector import ErrorInjector
from core.data_loader import LateBenchDataLoader
from utils.parallel import parallel_error_injection
from utils.storage import save_results
import argparse
import sys
from pathlib import Path

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_processing"))
from unified_schema import LateBenchExample


def validate_example_for_injection(example: LateBenchExample) -> bool:
    """Check if LateBenchExample is valid for error injection."""
    return len(example.solution.steps) >= 4  # Need at least 4 steps for late error injection


def main():
    parser = argparse.ArgumentParser(description='Run error injection on LateBench dataset')
    parser.add_argument('dataset', help='Dataset name (e.g., numinamath, prm800k)')
    parser.add_argument('--problem-type', default='all', choices=['all', 'complete', 'errors'],
                       help='Type of problems to process (default: all)')
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to process')
    parser.add_argument('--custom-suggestion', help='Custom error suggestion to use')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--max-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--output-name', help='Output experiment name (default: auto-generated)')
    
    args = parser.parse_args()
    
    print(f"🚀 LateBench Error Injection")
    print(f"Dataset: {args.dataset} ({args.problem_type})")
    print("=" * 60)
    
    # Load dataset
    print("📂 Loading dataset...")
    loader = LateBenchDataLoader()
    
    try:
        examples = loader.load_dataset(args.dataset, args.problem_type)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return 1
    
    # Limit examples if requested
    if args.max_examples and len(examples) > args.max_examples:
        examples = examples[:args.max_examples]
        print(f"📊 Limited to {len(examples)} examples")
    
    # Filter valid examples for injection
    print("🔄 Filtering examples for injection...")
    valid_examples = [example for example in examples if validate_example_for_injection(example)]
    
    print(f"✅ Found {len(valid_examples)} examples ready for injection")
    
    # Initialize error injector
    injector = ErrorInjector()
    
    # Run error injection
    print(f"🎲 Running error injection...")
    
    if args.parallel:
        print(f"   Using parallel processing with {args.max_workers} workers")
        
        def inject_with_suggestion(example):
            return injector.inject_error(example, args.custom_suggestion)
        
        # Create a mock injector object with the inject_error method
        mock_injector = type('MockInjector', (), {'inject_error': inject_with_suggestion})()
        
        results = parallel_error_injection(
            valid_examples, 
            mock_injector,
            max_workers=args.max_workers
        )
    else:
        print("   Using sequential processing")
        results = []
        
        for i, example in enumerate(valid_examples, 1):
            print(f"   Processing {i}/{len(valid_examples)}: {example.id}")
            
            # inject_error now modifies the example in place and returns it
            updated_example = injector.inject_error(example, args.custom_suggestion)
            results.append(updated_example)
            
            if updated_example.error_injection.success:
                print(f"   ✅ Success")
            else:
                error_msg = updated_example.error_injection.error_info.get('error', 'Unknown error')
                print(f"   ❌ Failed: {error_msg}")
    
    # Calculate statistics
    successful = sum(1 for r in results if r.error_injection.success)
    success_rate = successful / len(results) if results else 0
    
    print(f"\n📊 Injection Results:")
    print(f"   Total processed: {len(results)}")
    print(f"   Successful: {successful}")
    print(f"   Success rate: {success_rate:.1%}")
    
    # Save results
    output_name = args.output_name or f"error_injection_{args.dataset}"
    
    metadata = {
        'dataset': args.dataset,
        'problem_type': args.problem_type,
        'total_examples': len(results),
        'successful_injections': successful,
        'success_rate': success_rate,
        'custom_suggestion': args.custom_suggestion,
        'parallel_processing': args.parallel,
        'max_workers': args.max_workers if args.parallel else 1
    }
    
    # Convert LateBenchExample objects to dictionaries for storage
    results_dict = [example.to_dict() for example in results]
    output_path = save_results(results_dict, output_name, metadata)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"🎉 Error injection complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())