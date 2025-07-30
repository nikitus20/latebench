#!/usr/bin/env python3
"""
Create NuminaMath subsample dataset for manual annotation
Uses filtering logic to select high-quality problems from the full dataset
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.unified_schema import (
    LateBenchExample, LateBenchSource, LateBenchProblem, 
    LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
    LateBenchProcessing, generate_latebench_id, create_timestamp
)
from data_processing.numinamath_processor import NuminaMathProcessor
from utils.storage import save_examples_to_file

def get_filtering_criteria():
    """Get the filtering criteria for high-quality problems."""
    return {
        'min_steps': 20,          # Minimum number of solution steps (20+ as agreed)
        'max_steps': 50,          # Maximum number of solution steps
        'min_complexity': 3.0,    # Minimum complexity score
        'max_complexity': 7.0,    # Maximum complexity score
        'preferred_subjects': [   # Prefer these subjects
            'algebra', 'geometry', 'number_theory', 'calculus',
            'combinatorics', 'probability'
        ],
        'preferred_competitions': [  # Prefer these competition sources
            'olympiads', 'amc', 'aime'
        ]
    }

def calculate_selection_score(example) -> float:
    """Calculate selection score for an example based on filtering criteria."""
    criteria = get_filtering_criteria()
    score = 0.0
    
    # Step count score (prefer problems with 20+ steps)
    steps = len(example.solution.steps)
    if criteria['min_steps'] <= steps <= criteria['max_steps']:
        # Prefer problems with 20-35 steps (moderate complexity)
        if 20 <= steps <= 35:
            score += 2.0
        else:
            score += 1.0
    else:
        return 0.0  # Exclude if outside step range
    
    # Complexity score
    complexity = example.source.metadata.get('complexity_score', 3.0)
    if criteria['min_complexity'] <= complexity <= criteria['max_complexity']:
        # Prefer moderate complexity (4.0-6.0)
        if 4.0 <= complexity <= 6.0:
            score += 2.0
        else:
            score += 1.0
    else:
        return 0.0  # Exclude if outside complexity range
    
    # Subject preference
    subject = example.source.subject
    if subject in criteria['preferred_subjects']:
        score += 1.5
    
    # Competition preference
    competition = example.source.competition
    if competition in criteria['preferred_competitions']:
        score += 1.0
    
    # Boost problems that already have good metadata
    if example.source.metadata.get('estimated_steps', 0) > 0:
        score += 0.5
    
    # Add small random component to break ties
    score += random.random() * 0.1
    
    return score

def select_examples_for_dataset(all_examples: List, target_count: int = 500) -> List:
    """Select examples for the dataset using filtering criteria."""
    print(f"🔍 Filtering {len(all_examples)} examples...")
    
    # Calculate scores for all examples
    scored_examples = []
    for example in all_examples:
        score = calculate_selection_score(example)
        if score > 0:  # Only include examples that pass basic criteria
            scored_examples.append((score, example))
    
    print(f"📊 {len(scored_examples)} examples passed filtering criteria")
    
    # Sort by score (highest first) and take top examples
    scored_examples.sort(key=lambda x: x[0], reverse=True)
    selected_examples = [example for score, example in scored_examples[:target_count]]
    
    print(f"✅ Selected top {len(selected_examples)} examples")
    
    # Mark selected examples
    for example in selected_examples:
        example.source.metadata['selected_for_error_injection'] = True
    
    return selected_examples

def print_dataset_statistics(examples: List):
    """Print statistics about the selected dataset."""
    print(f"\n📊 DATASET STATISTICS")
    print(f"=" * 50)
    print(f"Total examples: {len(examples)}")
    
    # Step count distribution
    step_counts = [len(ex.solution.steps) for ex in examples]
    print(f"Steps - Min: {min(step_counts)}, Max: {max(step_counts)}, Avg: {sum(step_counts)/len(step_counts):.1f}")
    
    # Complexity distribution
    complexities = [ex.source.metadata.get('complexity_score', 3.0) for ex in examples]
    print(f"Complexity - Min: {min(complexities):.1f}, Max: {max(complexities):.1f}, Avg: {sum(complexities)/len(complexities):.1f}")
    
    # Subject distribution
    subjects = {}
    for ex in examples:
        subj = ex.source.subject
        subjects[subj] = subjects.get(subj, 0) + 1
    print(f"Subjects: {dict(sorted(subjects.items(), key=lambda x: x[1], reverse=True))}")
    
    # Competition distribution
    competitions = {}
    for ex in examples:
        comp = ex.source.competition or 'unknown'
        competitions[comp] = competitions.get(comp, 0) + 1
    print(f"Competitions: {dict(sorted(competitions.items(), key=lambda x: x[1], reverse=True))}")

def load_and_process_numinamath():
    """Load raw NuminaMath data and convert to LateBench format."""
    raw_file = "data/numinamath_full.json"
    
    if not Path(raw_file).exists():
        print(f"❌ Raw data file not found: {raw_file}")
        print("Please run: python scripts/download_numinamath.py")
        return []
    
    print(f"📂 Loading raw NuminaMath data from {raw_file}...")
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)
    
    print(f"✅ Loaded {len(raw_data)} raw examples")
    print("🔄 Converting to LateBench format...")
    
    # Initialize processor
    processor = NuminaMathProcessor()
    processed_examples = []
    
    for i, raw_example in enumerate(raw_data):
        if i % 10000 == 0:
            print(f"   Processed {i}/{len(raw_data)} examples...")
        
        try:
            # Convert to LateBench format
            latebench_example = processor.process_example(raw_example, i)
            if latebench_example:
                processed_examples.append(latebench_example)
        except Exception as e:
            # Skip problematic examples
            continue
    
    print(f"✅ Successfully converted {len(processed_examples)} examples to LateBench format")
    return processed_examples

def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(description="Create NuminaMath subsample dataset")
    
    parser.add_argument("--count", "-n", type=int, default=500,
                       help="Number of examples to select (default: 500)")
    parser.add_argument("--output", "-o", type=str,
                       help="Output filename (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible selection (default: 42)")
    
    return parser

def main():
    """Main function to create NuminaMath subsample dataset."""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    print(f"🚀 Creating {args.count}-problem NuminaMath subsample for manual annotation")
    
    # Set random seed for reproducible selection
    random.seed(args.seed)
    
    # Load and process all examples
    all_examples = load_and_process_numinamath()
    
    if not all_examples:
        print("❌ Failed to load and process dataset")
        return 1
    
    print(f"✅ Loaded {len(all_examples)} total examples")
    
    # Select examples using filtering criteria
    selected_examples = select_examples_for_dataset(all_examples, target_count=args.count)
    
    if len(selected_examples) < args.count:
        print(f"⚠️  Only found {len(selected_examples)} examples meeting criteria (requested {args.count})")
    
    # Print statistics
    print_dataset_statistics(selected_examples)
    
    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        output_file = f"data/datasets/latebench_numinamath_{len(selected_examples)}_for_annotation.json"
    
    print(f"\n💾 Saving dataset to {output_file}")
    save_examples_to_file(selected_examples, output_file)
    
    print(f"\n🎯 Dataset ready for manual annotation!")
    print(f"   - {len(selected_examples)} high-quality problems")
    print(f"   - Filtered by steps, complexity, subject, and competition")
    print(f"   - Random seed: {args.seed}")
    print(f"   - Saved to {output_file}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())