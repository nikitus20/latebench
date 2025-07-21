#!/usr/bin/env python3
"""
Download and analyze the NuminaMath dataset.
"""

import os
import sys
sys.path.append('src')

from data_loader import NuminaMathDataLoader

def main():
    # Create data loader
    loader = NuminaMathDataLoader()
    
    # Download dataset
    dataset = loader.download_dataset()
    
    if dataset is None:
        print("Failed to download dataset")
        return
    
    # Analyze dataset structure
    print("\nAnalyzing dataset structure...")
    analysis = loader.analyze_dataset_structure(dataset)
    
    print(f"\nDataset Analysis:")
    print(f"Total examples: {analysis['total_examples']}")
    print(f"Features: {analysis['features']}")
    
    if 'solution_statistics' in analysis:
        stats = analysis['solution_statistics']
        print(f"\nSolution Length Statistics:")
        print(f"  Mean: {stats['mean_length']:.2f} steps")
        print(f"  Median: {stats['median_length']:.2f} steps")
        print(f"  Min: {stats['min_length']} steps")
        print(f"  Max: {stats['max_length']} steps")
        print(f"  Std: {stats['std_length']:.2f} steps")
    
    # Filter for long solutions (8+ steps)
    print("\nFiltering for long solutions...")
    long_solutions = loader.filter_long_solutions(min_steps=8)
    
    # Save filtered dataset
    loader.save_filtered_dataset(long_solutions)
    
    print(f"\nFiltering complete. Found {len(long_solutions)} examples with 8+ steps")
    
    # Show a sample
    if long_solutions:
        print(f"\nSample example:")
        sample = long_solutions[0]
        print(f"Problem: {sample.get('problem', 'N/A')[:200]}...")
        print(f"Number of steps: {sample['num_steps']}")
        print(f"Answer: {sample.get('answer', 'N/A')}")

if __name__ == "__main__":
    main()