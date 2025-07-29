#!/usr/bin/env python3
"""
LateBench Dataset Filtering Script
Demonstrates how to use the filtering utilities to filter and analyze datasets
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.data_loader import LateBenchDataLoader
from utils.filtering import LateBenchFilter, LateBenchSorter, LateBenchSplitter, get_filtering_statistics
import json


def main():
    """Demonstrate filtering utilities"""
    
    print("🔧 LateBench Dataset Filtering Demo")
    print("=" * 50)
    
    # Load dataset
    loader = LateBenchDataLoader()
    print("\n📊 Available datasets:", list(loader.list_available_datasets().keys()))
    
    # Load NuminaMath as example
    examples = loader.load_dataset("numinamath", "complete")
    print(f"\n✅ Loaded {len(examples)} examples from NuminaMath")
    
    # Basic statistics
    print("\n📈 Basic Statistics:")
    stats = get_filtering_statistics(examples)
    print(f"   Total examples: {stats['total_examples']}")
    print(f"   Datasets: {list(stats['datasets'].keys())}")
    print(f"   Error status: {stats['error_status']}")
    print(f"   Step counts: min={stats['step_counts']['min']}, max={stats['step_counts']['max']}, avg={stats['step_counts']['avg']:.1f}")
    
    # Filtering examples
    print("\n🔍 Filtering Examples:")
    
    # Filter by step count
    long_examples = LateBenchFilter.by_step_count(examples, min_steps=25)
    print(f"   Examples with ≥25 steps: {len(long_examples)}")
    
    # Filter by difficulty
    medium_examples = LateBenchFilter.by_difficulty(examples, min_difficulty=2.5, max_difficulty=4.0)
    print(f"   Medium difficulty examples: {len(medium_examples)}")
    
    # Filter by dataset
    numinamath_examples = LateBenchFilter.by_dataset(examples, ["numinamath"])
    print(f"   NuminaMath examples: {len(numinamath_examples)}")
    
    # Sorting examples
    print("\n🔄 Sorting Examples:")
    sorted_by_steps = LateBenchSorter.by_step_count(examples, reverse=True)
    print(f"   Longest solution: {sorted_by_steps[0].solution.total_steps} steps")
    print(f"   Shortest solution: {sorted_by_steps[-1].solution.total_steps} steps")
    
    # Splitting examples
    print("\n📂 Splitting Examples:")
    splits = LateBenchSplitter.by_error_status(examples)
    print(f"   Complete solutions: {len(splits['complete'])}")
    print(f"   Error solutions: {len(splits['errors'])}")
    
    # Train/test split
    train_test = LateBenchSplitter.train_test_split(examples, test_ratio=0.2, random_seed=42)
    print(f"   Train set: {len(train_test['train'])}")
    print(f"   Test set: {len(train_test['test'])}")
    
    # Custom filtering example
    print("\n🎯 Custom Filtering Example:")
    complex_examples = LateBenchFilter.by_custom_criteria(
        examples, 
        lambda ex: ex.solution.total_steps > 20 and ex.source.subject == "mathematics"
    )
    print(f"   Complex math examples (>20 steps): {len(complex_examples)}")
    
    print("\n✅ Filtering demo completed!")


if __name__ == "__main__":
    main()