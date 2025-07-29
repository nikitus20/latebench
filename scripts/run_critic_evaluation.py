#!/usr/bin/env python3
"""
Unified critic evaluation script for LateBench.
Evaluate LateBenchExample objects for errors using LLM critic.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.critic import MathCritic
from core.data_loader import LateBenchDataLoader
from core.metrics import DeltaBenchEvaluator, print_metrics_summary
from utils.parallel import parallel_critic_evaluation
from utils.storage import save_results, load_results
import argparse
import json

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "data_processing"))
from unified_schema import LateBenchExample


def main():
    parser = argparse.ArgumentParser(description='Run critic evaluation on LateBench dataset')
    parser.add_argument('input', help='Dataset name or path to injection results')
    parser.add_argument('--problem-type', default='all', choices=['all', 'complete', 'errors'],
                       help='Type of problems to evaluate (for datasets)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Critic model to use')
    parser.add_argument('--max-examples', type=int, help='Maximum number of examples to evaluate')
    parser.add_argument('--parallel', action='store_true', help='Use parallel processing')
    parser.add_argument('--max-workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--compute-metrics', action='store_true', help='Compute DeltaBench metrics')
    parser.add_argument('--output-name', help='Output experiment name')
    
    args = parser.parse_args()
    
    print(f"🔍 LateBench Critic Evaluation")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print("=" * 60)
    
    # Determine if input is dataset or results file
    examples = []
    
    if os.path.exists(args.input):
        # Load from results file (containing LateBenchExample objects with injected errors)
        print("📂 Loading from results file...")
        try:
            results_data, metadata = load_results(args.input)
            
            # Convert results back to LateBenchExample objects
            for result_dict in results_data:
                try:
                    example = LateBenchExample.from_dict(result_dict)
                    examples.append(example)
                except Exception as e:
                    print(f"Warning: Failed to parse result, skipping: {e}")
            
            print(f"✅ Loaded {len(examples)} examples from results")
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return 1
    
    else:
        # Load from dataset
        print("📂 Loading dataset...")
        loader = LateBenchDataLoader()
        
        try:
            examples = loader.load_dataset(args.input, args.problem_type)
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return 1
    
    # Limit examples if requested
    if args.max_examples and len(examples) > args.max_examples:
        examples = examples[:args.max_examples]
        print(f"📊 Limited to {len(examples)} examples")
    
    print(f"✅ Prepared {len(examples)} examples for evaluation")
    
    # Initialize critic
    critic = MathCritic(model=args.model)
    
    # Run critic evaluation
    print(f"🔍 Running critic evaluation...")
    
    if args.parallel:
        print(f"   Using parallel processing with {args.max_workers} workers")
        critic_results = critic.evaluate_batch(examples, max_workers=args.max_workers)
    else:
        print("   Using sequential processing")
        critic_results = []
        
        for i, example in enumerate(examples, 1):
            print(f"   Evaluating {i}/{len(examples)}: {example.id}")
            
            result = critic.evaluate_example(example)
            critic_results.append(result)
            
            status = "errors found" if result.has_errors else "no errors"
            print(f"   ✅ {status}")
    
    # Calculate statistics
    total_evaluated = len(critic_results)
    found_errors = sum(1 for r in critic_results if r.has_errors)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Total evaluated: {total_evaluated}")
    print(f"   Problems with errors found: {found_errors}")
    print(f"   Error detection rate: {found_errors/total_evaluated:.1%}")
    
    # Compute DeltaBench metrics if requested
    deltabench_metrics = None
    if args.compute_metrics:
        print(f"\n📈 Computing DeltaBench metrics...")
        
        evaluator = DeltaBenchEvaluator(use_first_error_cutoff=True)
        
        # Create critic results dict
        critic_dict = {}
        for i, result in enumerate(critic_results):
            critic_dict[examples[i].id] = result
        
        try:
            deltabench_metrics = evaluator.evaluate_batch(examples, critic_dict)
            print_metrics_summary(deltabench_metrics)
        except Exception as e:
            print(f"❌ Error computing metrics: {e}")
    
    # Save results
    output_name = args.output_name or f"critic_evaluation_{args.model}"
    
    metadata = {
        'input_source': args.input,
        'critic_model': args.model,
        'total_evaluated': total_evaluated,
        'errors_found': found_errors,
        'error_detection_rate': found_errors / total_evaluated if total_evaluated > 0 else 0,
        'parallel_processing': args.parallel,
        'max_workers': args.max_workers if args.parallel else 1,
        'deltabench_metrics': deltabench_metrics.to_dict() if deltabench_metrics else None
    }
    
    # Prepare results for storage
    storage_results = []
    for i, result in enumerate(critic_results):
        storage_result = result.to_dict()
        storage_result['example_id'] = examples[i].id
        storage_results.append(storage_result)
    
    output_path = save_results(storage_results, output_name, metadata)
    
    print(f"\n💾 Results saved to: {output_path}")
    print(f"🎉 Critic evaluation complete!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())