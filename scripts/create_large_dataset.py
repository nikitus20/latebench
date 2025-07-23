#!/usr/bin/env python3
"""
Create large-scale dataset with 500 problems of 25+ steps for error injection.
"""

import sys
sys.path.append('src')

import json
import random
import os
from datetime import datetime
from error_injector import AdversarialErrorInjector

def filter_complex_problems():
    """Filter and select 500 problems with 25+ steps."""
    print("Loading filtered dataset...")
    
    with open('data/filtered_long_solutions.json', 'r') as f:
        all_problems = json.load(f)
    
    print(f"Total problems loaded: {len(all_problems):,}")
    
    # Filter for 25+ step problems
    complex_problems = []
    
    print("Filtering for 25+ step problems...")
    injector = AdversarialErrorInjector()
    
    for i, problem in enumerate(all_problems):
        if i % 10000 == 0:
            print(f"  Processed {i:,}/{len(all_problems):,} problems...")
        
        # Count steps
        if 'num_steps' in problem:
            num_steps = problem['num_steps']
        elif 'parsed_steps' in problem:
            num_steps = len(problem['parsed_steps'])
        else:
            steps = injector.parse_solution_steps(problem.get('solution', ''))
            num_steps = len(steps)
            problem['num_steps'] = num_steps  # Cache for future use
        
        if num_steps >= 25:
            complex_problems.append(problem)
    
    print(f"Found {len(complex_problems):,} problems with 25+ steps")
    
    # Randomly select 500 problems
    if len(complex_problems) >= 500:
        selected_problems = random.sample(complex_problems, 500)
        print(f"Randomly selected 500 problems from {len(complex_problems):,} available")
    else:
        selected_problems = complex_problems
        print(f"Using all {len(complex_problems)} available problems (less than 500)")
    
    # Add metadata
    for i, problem in enumerate(selected_problems):
        problem['dataset_index'] = i
        problem['selection_timestamp'] = datetime.now().isoformat()
        problem['error_placement'] = 'last_33_percent'
    
    return selected_problems

def organize_data_structure():
    """Create organized data structure for large-scale processing."""
    
    # Create directory structure
    experiment_name = f"large_scale_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    base_dir = f"data/experiments/{experiment_name}"
    
    directories = [
        base_dir,
        f"{base_dir}/selected_problems",
        f"{base_dir}/results", 
        f"{base_dir}/checkpoints",
        f"{base_dir}/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return base_dir, experiment_name

def main():
    """Main execution function."""
    print("🚀 Creating Large-Scale LateBench Dataset")
    print("=" * 50)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Organize data structure
    base_dir, experiment_name = organize_data_structure()
    
    # Filter complex problems
    selected_problems = filter_complex_problems()
    
    # Save selected problems
    selected_path = f"{base_dir}/selected_problems/complex_25plus_steps.json"
    with open(selected_path, 'w') as f:
        json.dump(selected_problems, f, indent=2)
    
    print(f"✅ Saved {len(selected_problems)} selected problems to: {selected_path}")
    
    # Create experiment metadata
    metadata = {
        "experiment_name": experiment_name,
        "created_timestamp": datetime.now().isoformat(),
        "total_problems": len(selected_problems),
        "min_steps": 25,
        "error_placement": "last_33_percent",
        "selection_method": "random_sample",
        "random_seed": 42,
        "model": "gpt-4-turbo-preview",
        "base_directory": base_dir
    }
    
    metadata_path = f"{base_dir}/experiment_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved experiment metadata to: {metadata_path}")
    
    # Analyze selected problems
    step_counts = [p.get('num_steps', 0) for p in selected_problems]
    avg_steps = sum(step_counts) / len(step_counts)
    max_steps = max(step_counts)
    min_steps = min(step_counts)
    
    print(f"\\n📊 Selected Dataset Analysis:")
    print(f"  Total problems: {len(selected_problems)}")
    print(f"  Step range: {min_steps} - {max_steps}")
    print(f"  Average steps: {avg_steps:.1f}")
    
    # Step distribution
    ranges = {
        "25-30 steps": len([s for s in step_counts if 25 <= s <= 30]),
        "31-40 steps": len([s for s in step_counts if 31 <= s <= 40]),
        "41-50 steps": len([s for s in step_counts if 41 <= s <= 50]),
        "50+ steps": len([s for s in step_counts if s > 50])
    }
    
    print(f"\\n  Step distribution:")
    for range_name, count in ranges.items():
        percentage = 100 * count / len(selected_problems)
        print(f"    {range_name}: {count} ({percentage:.1f}%)")
    
    print(f"\\n🎯 Ready for large-scale error injection!")
    print(f"   Experiment: {experiment_name}")
    print(f"   Base directory: {base_dir}")
    print(f"   Next step: Run error injection with checkpointing")
    
    return base_dir, experiment_name

if __name__ == "__main__":
    main()