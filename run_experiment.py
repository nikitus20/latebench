#!/usr/bin/env python3
"""
Run initial experiments with LateBench.
"""

import sys
sys.path.append('src')

import os
import argparse
from data_loader import NuminaMathDataLoader
from error_injector import AdversarialErrorInjector
from error_types import MATH_ERROR_TAXONOMY
from visualization import VISUALIZER

def run_small_experiment(num_examples=5):
    """Run a small experiment with a few examples."""
    print(f"=== Running Small Experiment ({num_examples} examples) ===\n")
    
    # Initialize components
    loader = NuminaMathDataLoader(cache_dir="./data")
    injector = AdversarialErrorInjector()
    
    # Load data
    print("Loading sample problems...")
    try:
        # Try to load from filtered cache first
        problems = loader.load_filtered_dataset()
        if not problems:
            print("No filtered dataset found, creating one...")
            dataset = loader.download_dataset()
            problems = loader.filter_long_solutions(min_steps=8)
            loader.save_filtered_dataset(problems)
        
        print(f"Found {len(problems)} suitable problems")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return False
    
    if len(problems) < num_examples:
        print(f"Only {len(problems)} problems available, using all")
        num_examples = len(problems)
    
    # Select sample
    sample_problems = problems[:num_examples]
    print(f"Selected {len(sample_problems)} problems for experiment")
    
    # Define error distribution for variety
    error_distribution = {
        "invalid_generalization": 0.25,
        "theorem_misapplication": 0.25,
        "circular_reasoning": 0.25,
        "domain_restriction_violation": 0.25
    }
    
    print(f"Error distribution: {error_distribution}")
    
    # Run batch injection
    print("\nStarting error injection...")
    results = injector.batch_inject_errors(
        sample_problems,
        error_distribution=error_distribution,
        save_checkpoints=True,
        checkpoint_interval=2
    )
    
    # Save results
    results_path = "./data/small_experiment_results.json"
    injector.save_results(results, results_path)
    
    # Generate analysis
    successful_results = [r for r in results if r.success]
    success_rate = len(successful_results) / len(results)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Success rate: {success_rate:.1%} ({len(successful_results)}/{len(results)})")
    
    if successful_results:
        # Show sample result
        sample_result = successful_results[0]
        print(f"\nSample successful injection:")
        print(f"  Error type: {sample_result.error_analysis.get('error_type')}")
        print(f"  Error step: {sample_result.error_analysis.get('selected_error_step')}")
        print(f"  Original answer: {sample_result.original_problem.get('answer')}")
        print(f"  Modified answer: {sample_result.modified_solution.get('final_answer')}")
        
        # Generate quality metrics
        metrics = VISUALIZER.create_quality_metrics_report(results)
        print(f"\nQuality Metrics:")
        print(f"  Unique error types used: {metrics['error_type_metrics']['diversity_score']:.1%}")
        print(f"  Late position compliance: {metrics['position_metrics']['last_quarter_compliance']:.1%}")
        print(f"  High difficulty rate: {metrics['difficulty_metrics']['high_difficulty_rate']:.1%}")
        
        # Generate visualizations
        try:
            print("\nGenerating visualizations...")
            VISUALIZER.create_batch_statistics_plot(results, save_path='./data/experiment_stats.png')
            VISUALIZER.save_html_report(results, './data/experiment_report.html')
            print("✓ Visualizations saved")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
        
        # Save sample visualization
        try:
            viz = VISUALIZER.create_example_visualization(successful_results[0])
            with open('./data/sample_injection.md', 'w') as f:
                f.write(viz)
            print("✓ Sample injection saved to ./data/sample_injection.md")
        except Exception as e:
            print(f"Error creating sample visualization: {e}")
    
    return True

def run_full_experiment(num_examples=100):
    """Run a full experiment with more examples."""
    print(f"=== Running Full Experiment ({num_examples} examples) ===\n")
    
    # Similar to small experiment but with more examples and balanced distribution
    loader = NuminaMathDataLoader(cache_dir="./data")
    injector = AdversarialErrorInjector()
    
    print("Loading problems...")
    problems = loader.load_filtered_dataset()
    if not problems:
        print("No filtered dataset found. Please run download_data.py first.")
        return False
    
    print(f"Found {len(problems)} suitable problems")
    
    if len(problems) < num_examples:
        print(f"Only {len(problems)} problems available, using all")
        num_examples = len(problems)
    
    # Use balanced distribution across all error types
    all_errors = MATH_ERROR_TAXONOMY.get_all_error_names()
    error_distribution = {error: 1.0/len(all_errors) for error in all_errors}
    
    # Sample problems
    import random
    random.seed(42)  # For reproducibility
    sample_problems = random.sample(problems, num_examples)
    
    print(f"Selected {len(sample_problems)} problems")
    print(f"Using balanced distribution across {len(all_errors)} error types")
    
    # Run experiment
    results = injector.batch_inject_errors(
        sample_problems,
        error_distribution=error_distribution,
        save_checkpoints=True,
        checkpoint_interval=10
    )
    
    # Save results
    results_path = "./data/full_experiment_results.json"
    injector.save_results(results, results_path)
    
    # Analysis
    successful_results = [r for r in results if r.success]
    success_rate = len(successful_results) / len(results)
    
    print(f"\n=== Full Experiment Complete ===")
    print(f"Success rate: {success_rate:.1%} ({len(successful_results)}/{len(results)})")
    
    if successful_results:
        # Generate comprehensive analysis
        metrics = VISUALIZER.create_quality_metrics_report(results)
        
        print(f"\nComprehensive Quality Metrics:")
        print(f"  Error type diversity: {metrics['error_type_metrics']['diversity_score']:.1%}")
        print(f"  Late position compliance: {metrics['position_metrics']['last_quarter_compliance']:.1%}")
        print(f"  High difficulty rate: {metrics['difficulty_metrics']['high_difficulty_rate']:.1%}")
        print(f"  Mean relative position: {metrics['position_metrics']['mean_relative_position']:.2f}")
        
        # Error type breakdown
        print(f"\nError Type Distribution:")
        for error_type, count in metrics['error_type_metrics']['distribution'].items():
            print(f"  {error_type}: {count}")
        
        # Generate all visualizations
        try:
            print("\nGenerating comprehensive visualizations...")
            VISUALIZER.create_batch_statistics_plot(results, save_path='./data/full_experiment_stats.png')
            VISUALIZER.create_error_type_analysis(results, save_path='./data/full_experiment_analysis.png')
            VISUALIZER.save_html_report(results, './data/full_experiment_report.html', max_examples=10)
            print("✓ All visualizations saved")
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Run LateBench experiments')
    parser.add_argument('--experiment', choices=['small', 'full'], default='small',
                       help='Type of experiment to run')
    parser.add_argument('--num_examples', type=int, default=None,
                       help='Number of examples to process')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found in environment")
        print("Please add your OpenAI API key to the .env file")
        return 1
    
    if not os.path.exists("./data/numinamath_local"):
        print("❌ Dataset not found")
        print("Please run: python download_data.py first")
        return 1
    
    # Run experiment
    try:
        if args.experiment == 'small':
            num_examples = args.num_examples or 5
            success = run_small_experiment(num_examples)
        else:  # full
            num_examples = args.num_examples or 100
            success = run_full_experiment(num_examples)
        
        if success:
            print("\n✅ Experiment completed successfully!")
            print("\nGenerated files:")
            print("  - ./data/*_results.json (raw results)")
            print("  - ./data/*_report.html (interactive report)")
            print("  - ./data/*_stats.png (statistical plots)")
            return 0
        else:
            print("\n❌ Experiment failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Experiment interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())