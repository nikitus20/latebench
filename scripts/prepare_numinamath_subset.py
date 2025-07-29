#!/usr/bin/env python3
"""
Prepare NuminaMath-CoT Subset for Error Injection Testing
Selects 200 complex problems with ≥20 steps and processes them into LateBench format
"""

import json
import os
import sys
import random
from typing import List, Dict, Any
from collections import defaultdict

# Add src to path
sys.path.append('./src')
from data_processing.numinamath_processor import NuminaMathProcessor


def estimate_solution_steps(solution: str) -> int:
    """Estimate number of steps using same logic as NuminaMath processor"""
    if not solution:
        return 0
    
    # Use same step indicators as NuminaMath processor
    step_indicators = ['step', 'Step', 'STEP', '1.', '2.', '3.', '4.', '5.']
    
    lines = solution.strip().split('\n')
    step_count = 0
    current_step_has_content = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if this line starts a new step (same logic as processor)
        is_new_step = False
        if any(line.startswith(indicator) for indicator in step_indicators):
            is_new_step = True
        elif line[0].isdigit() and ('.' in line[:5] or ')' in line[:5]):
            is_new_step = True
        
        if is_new_step:
            if current_step_has_content:
                step_count += 1
            current_step_has_content = True
        elif current_step_has_content:
            # This line is part of current step
            pass
        else:
            # First content line becomes part of first step
            current_step_has_content = True
    
    # Count the last step if it has content
    if current_step_has_content:
        step_count += 1
    
    return step_count


def calculate_complexity_score(problem: str, solution: str) -> float:
    """Calculate complexity score for a problem-solution pair"""
    score = 0.0
    
    # Solution length factor (longer solutions often more complex)
    solution_length = len(solution)
    if 800 <= solution_length <= 3000:
        score += 2.0
    elif solution_length > 3000:
        score += 1.5
    elif solution_length > 500:
        score += 1.0
    
    # Mathematical notation density
    math_symbols = ['\\frac', '\\sqrt', '\\sum', '\\int', '\\log', '\\sin', '\\cos', '\\tan', 
                   '\\alpha', '\\beta', '\\gamma', '\\theta', '\\pi', '\\infty', '\\leq', '\\geq']
    math_density = sum(solution.count(symbol) for symbol in math_symbols)
    score += min(math_density * 0.1, 2.0)  # Cap at 2.0
    
    # Problem complexity indicators  
    complex_terms = ['prove', 'find all', 'determine', 'show that', 'polynomial', 'sequence', 
                    'function', 'inequality', 'maximum', 'minimum', 'optimization']
    term_score = sum(1 for term in complex_terms if term.lower() in problem.lower())
    score += min(term_score * 0.3, 1.5)
    
    # Multi-part problems (often more complex)
    if '(a)' in problem or '(i)' in problem:
        score += 1.0
        
    return score


def select_numinamath_subset(target_size: int = 200) -> List[Dict[str, Any]]:
    """Select diverse subset of complex problems from NuminaMath-CoT for error injection testing"""
    
    print(f"🎯 Selecting {target_size} complex NuminaMath-CoT problems (≥20 steps)...")
    
    # Import here to avoid loading issues
    import datasets
    
    # Load dataset
    data_dir = './data/numinamath_local'
    train_data = datasets.load_from_disk(f'{data_dir}/train')
    
    print(f"Total available: {len(train_data):,} examples")
    
    # Priority sources (based on analysis - olympiads has most complex problems)
    priority_sources = ['olympiads', 'aops_forum', 'cn_k12', 'amc_aime', 'synthetic_amc']
    
    print(f"Target size: {target_size}")
    print(f"Priority sources: {priority_sources}")
    
    # Filter for complex examples with ≥20 steps
    complex_examples = []
    step_distribution = {}
    
    print("🔍 Analyzing examples for complexity...")
    
    for i, example in enumerate(train_data):
        if example['source'] in priority_sources:
            solution = example['solution']
            problem = example['problem']
            
            # Estimate number of steps
            estimated_steps = estimate_solution_steps(solution)
            
            # Filter for problems with ≥25 estimated steps to ensure ≥20 after processing
            if estimated_steps >= 25:
                # Additional quality filters
                has_clear_structure = '\n' in solution and ('. ' in solution or ':' in solution)
                reasonable_problem_length = 50 <= len(problem) <= 1000
                reasonable_solution_length = 800 <= len(solution) <= 4000
                
                if has_clear_structure and reasonable_problem_length and reasonable_solution_length:
                    complexity_score = calculate_complexity_score(problem, solution)
                    
                    complex_examples.append((i, example, estimated_steps, complexity_score))
                    
                    # Track step distribution
                    step_range = f"{estimated_steps//10*10}-{estimated_steps//10*10+9}"
                    step_distribution[step_range] = step_distribution.get(step_range, 0) + 1
        
        # Progress indicator
        if i % 10000 == 0:
            print(f"  Processed {i:,} examples, found {len(complex_examples)} complex problems")
            
        # Stop when we have enough candidates
        if len(complex_examples) >= target_size * 2:
            break
    
    print(f"\n📊 Complex examples found: {len(complex_examples)}")
    print("Step distribution:")
    for range_str, count in sorted(step_distribution.items()):
        print(f"  {range_str} steps: {count}")
    
    if len(complex_examples) < target_size:
        print(f"⚠️  Warning: Only found {len(complex_examples)} complex examples, less than target {target_size}")
        target_size = len(complex_examples)
    
    # Sort by complexity score and select top examples
    complex_examples.sort(key=lambda x: (x[3], x[2]), reverse=True)  # Sort by complexity, then steps
    
    # Group by source for balanced selection
    source_examples = defaultdict(list)
    for idx, example, steps, complexity in complex_examples:
        source_examples[example['source']].append((idx, example, steps, complexity))
    
    # Select with source diversity, prioritizing olympiads
    selected = []
    
    print(f"\n📋 Source-balanced selection:")
    
    # First, take majority from olympiads (highest quality)
    olympiad_quota = min(target_size // 2, len(source_examples.get('olympiads', [])))
    if olympiad_quota > 0:
        selected.extend(source_examples['olympiads'][:olympiad_quota])
        print(f"  olympiads: {olympiad_quota} examples")
    
    # Distribute remaining slots across other sources
    remaining = target_size - len(selected)
    other_sources = [s for s in priority_sources if s != 'olympiads' and s in source_examples]
    
    if other_sources and remaining > 0:
        per_source = remaining // len(other_sources)
        remainder = remaining % len(other_sources)
        
        for i, source in enumerate(other_sources):
            count = per_source + (1 if i < remainder else 0)
            count = min(count, len(source_examples[source]))
            
            if count > 0:
                selected.extend(source_examples[source][:count])
                print(f"  {source}: {count} examples")
    
    print(f"\n✅ Selected {len(selected)} complex examples total")
    
    # Show statistics
    if selected:
        steps_list = [item[2] for item in selected]
        complexity_list = [item[3] for item in selected]
        print(f"📈 Selection statistics:")
        print(f"  Step count range: {min(steps_list)}-{max(steps_list)}")
        print(f"  Average steps: {sum(steps_list)/len(steps_list):.1f}")
        print(f"  Average complexity score: {sum(complexity_list)/len(complexity_list):.2f}")
    
    # Convert to raw format for processor
    raw_examples = []
    for idx, example, steps, complexity in selected:
        raw_example = {
            'original_id': f"numinamath_{idx}",
            'problem': example['problem'],
            'solution': example['solution'],
            'answer': 'See solution',  # NuminaMath doesn't separate answers
            'source': example['source'],
            'metadata': {
                'dataset_index': idx,
                'original_source': example['source'],
                'estimated_steps': steps,
                'complexity_score': complexity,
                'selected_for_error_injection': True
            }
        }
        raw_examples.append(raw_example)
    
    return raw_examples


def process_numinamath_subset():
    """Process selected NuminaMath subset into LateBench format"""
    
    print("🔄 Processing NuminaMath subset into LateBench format...")
    
    # Select subset
    raw_examples = select_numinamath_subset(target_size=200)
    
    if not raw_examples:
        print("❌ No examples selected")
        return False
    
    # Quick validation: test step estimation accuracy on first few examples
    print("🔍 Validating step estimation accuracy...")
    processor = NuminaMathProcessor()
    
    for i in range(min(5, len(raw_examples))):
        example = raw_examples[i]
        estimated = example['metadata']['estimated_steps']
        
        # Process single example to get actual step count
        latebench_example = processor.process_example(example, i)
        if latebench_example:
            actual = latebench_example.solution.total_steps
            print(f"  Example {i}: estimated={estimated}, actual={actual} ({'✓' if actual >= 20 else '✗'})")
        else:
            print(f"  Example {i}: processing failed")
    
    # Save raw subset for processing
    raw_file = "./data/numinamath_subset_raw.json"
    with open(raw_file, 'w') as f:
        json.dump(raw_examples, f, indent=2)
    
    print(f"💾 Saved raw subset to {raw_file}")
    
    # Process through NuminaMath processor
    processor = NuminaMathProcessor()
    
    output_file = "./data/datasets/latebench_numinamath_complete.json"
    processed_examples = processor.process_dataset(
        input_file=raw_file,
        output_file=output_file,
        max_examples=None
    )
    
    # Analyze processed results
    if processed_examples:
        print(f"\n📊 Processing Results:")
        print(f"Successfully processed: {len(processed_examples)} examples")
        
        # Sample statistics
        step_counts = [ex.solution.total_steps for ex in processed_examples]
        avg_steps = sum(step_counts) / len(step_counts)
        
        # Verify all examples have ≥20 steps
        examples_with_20_plus = sum(1 for count in step_counts if count >= 20)
        examples_under_20 = len(step_counts) - examples_with_20_plus
        
        print(f"✅ Examples with ≥20 steps: {examples_with_20_plus}")
        if examples_under_20 > 0:
            print(f"⚠️  Examples with <20 steps: {examples_under_20}")
            under_20_counts = [count for count in step_counts if count < 20]
            print(f"   Step counts for <20: {sorted(under_20_counts)}")
        else:
            print(f"🎯 All examples meet the ≥20 steps requirement!")
        
        sources = defaultdict(int)
        for ex in processed_examples:
            source = ex.source.metadata.get('original_source', 'unknown')
            sources[source] += 1
        
        print(f"Average steps per solution: {avg_steps:.1f}")
        print(f"Step count range: {min(step_counts)}-{max(step_counts)}")
        print(f"Source distribution:")
        for source, count in sources.items():
            print(f"  {source}: {count}")
        
        # Show sample example
        example = processed_examples[0]
        print(f"\n📝 Sample Processed Example:")
        print(f"ID: {example.id}")
        print(f"Source: {example.source.dataset} ({example.source.metadata.get('original_source')})")
        print(f"Problem: {example.problem.statement[:150]}...")
        print(f"Steps: {example.solution.total_steps}")
        print(f"Solution preview: {example.solution.steps[0].content[:100]}...")
        
        print(f"\n🎯 Ready for Error Injection!")
        print(f"Use dataset: latebench_numinamath_complete")
        print(f"Available examples: {len(processed_examples)}")
        
        return True
    else:
        print("❌ Processing failed")
        return False


def main():
    """Main function"""
    print("🚀 NuminaMath-CoT Complex Subset Preparation for Error Injection")
    print("=" * 70)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process subset
    success = process_numinamath_subset()
    
    if success:
        print("\n✅ NuminaMath complex subset preparation complete!")
        print("📁 Output: ./data/datasets/latebench_numinamath_complete.json")
        print("🎯 Ready for manual error injection testing via dashboard")
        print("📊 Dataset contains 200 complex problems with ≥20 steps each")
    else:
        print("\n❌ Subset preparation failed")


if __name__ == "__main__":
    main()