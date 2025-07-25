#!/usr/bin/env python3
"""
Filter PRM800K Dataset for MATH Level 5 Problems with Natural Late Errors

This script maps PRM800K problems back to their original MATH dataset difficulty levels
and filters for level 5 problems where natural human-annotated errors appear at step 12 
or later. The resulting dataset contains problems with naturally occurring errors from 
the PRM800K human annotations, marked as "natural" errors to distinguish from future 
artificially injected errors.
"""

import json
import re
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

def normalize_problem_text(text: str) -> str:
    """Normalize problem text for reliable matching"""
    # Remove extra whitespace and normalize LaTeX formatting
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\$\s+', '$', text)
    text = re.sub(r'\s+\$', '$', text)
    return text

def load_math_dataset_index() -> Dict[str, Dict]:
    """Load MATH dataset and create index by problem text"""
    print("🔍 Loading MATH dataset index...")
    
    math_index = {}
    train_file = './data/sources/prm800k/prm800k/math_splits/train.jsonl'
    test_file = './data/sources/prm800k/prm800k/math_splits/test.jsonl'
    
    # Load train data
    train_count = 0
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    normalized_problem = normalize_problem_text(data['problem'])
                    math_index[normalized_problem] = {
                        'level': data['level'],
                        'subject': data['subject'],
                        'unique_id': data['unique_id'],
                        'answer': data['answer'],
                        'solution': data['solution'],
                        'split': 'train'
                    }
                    train_count += 1
    
    # Load test data
    test_count = 0
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    normalized_problem = normalize_problem_text(data['problem'])
                    math_index[normalized_problem] = {
                        'level': data['level'],
                        'subject': data['subject'],
                        'unique_id': data['unique_id'],
                        'answer': data['answer'],
                        'solution': data['solution'],
                        'split': 'test'
                    }
                    test_count += 1
    
    print(f"✅ Loaded MATH index: {train_count} train + {test_count} test = {len(math_index)} unique problems")
    
    # Show level distribution
    level_counts = defaultdict(int)
    for entry in math_index.values():
        level_counts[entry['level']] += 1
    
    print("📊 MATH Dataset Level Distribution:")
    for level in sorted(level_counts.keys()):
        print(f"   Level {level}: {level_counts[level]} problems")
    
    return math_index

def find_first_error_step(label_data: Dict) -> Optional[int]:
    """Find the step number where the first error occurs (rating == -1)"""
    steps = label_data.get('steps', [])
    
    for i, step in enumerate(steps):
        completions = step.get('completions', [])
        if completions:
            rating = completions[0].get('rating', 1)
            if rating == -1:  # Error step
                return i + 1  # Convert to 1-based indexing
    
    return None  # No error found (complete solution)

def filter_prm800k_level5_late_errors(max_examples: int = 200) -> List[Dict]:
    """Filter PRM800K for MATH level 5 problems with errors at step 12+"""
    print("🚀 Filtering PRM800K for Level 5 problems with late error injection...")
    
    # Load MATH dataset index
    math_index = load_math_dataset_index()
    
    # Load PRM800K Phase 2 data
    phase2_file = './data/sources/prm800k/prm800k/data/phase2_train.jsonl'
    
    if not os.path.exists(phase2_file):
        print(f"❌ PRM800K Phase2 file not found: {phase2_file}")
        return []
    
    print(f"📖 Processing PRM800K Phase 2 data from {phase2_file}")
    
    filtered_examples = []
    total_processed = 0
    level5_found = 0
    late_error_found = 0
    mapping_failures = 0
    
    with open(phase2_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            if len(filtered_examples) >= max_examples:
                break
                
            total_processed += 1
            if total_processed % 1000 == 0:
                print(f"   Processed {total_processed} examples, found {len(filtered_examples)} matches")
            
            try:
                prm_data = json.loads(line)
                
                # Extract problem text from PRM800K
                problem_text = prm_data['question']['problem']
                normalized_problem = normalize_problem_text(problem_text)
                
                # Map to MATH dataset
                if normalized_problem not in math_index:
                    mapping_failures += 1
                    continue
                
                math_info = math_index[normalized_problem]
                
                # Filter for Level 5 problems
                if math_info['level'] != 5:
                    continue
                
                level5_found += 1
                
                # Find first error step
                first_error_step = find_first_error_step(prm_data['label'])
                
                # Skip if no error found or error is too early
                if first_error_step is None or first_error_step < 12:
                    continue
                
                late_error_found += 1
                
                # Create filtered example
                filtered_example = {
                    'id': f"lb_prm800k_level5_{len(filtered_examples):03d}",
                    'source': {
                        'dataset': 'prm800k',
                        'original_id': f"prm_phase2_{total_processed}",
                        'difficulty': 5,  # Original MATH level
                        'subject': math_info['subject'].lower(),
                        'competition': 'Math Competition',
                        'metadata': {
                            'finish_reason': prm_data['label'].get('finish_reason', ''),
                            'human_steps': len(prm_data['label'].get('steps', [])),
                            'has_errors': True,
                            'first_error_step': first_error_step,
                            'error_source': 'natural',
                            'error_origin': 'prm800k_human_annotation',
                            'error_type': 'human_annotated',
                            'is_filtered_dataset': True,
                            'filter_criteria': 'level5_late_errors_step12plus',
                            'math_unique_id': math_info['unique_id'],
                            'math_split': math_info['split']
                        }
                    },
                    'problem': {
                        'statement': problem_text
                    },
                    'solution': {
                        'steps': [],  # Will be populated from human annotations
                        'final_answer': prm_data['question'].get('ground_truth_answer', ''),
                        'total_steps': len(prm_data['label'].get('steps', [])),
                        'solution_method': 'analytical'
                    },
                    'error_injection': {
                        'has_errors': True,
                        'error_info': None,
                        'manual_attempts': [],
                        'final_decision': None,
                        'decision_timestamp': None,
                        'custom_suggestions': []
                    },
                    'processing': {
                        'added_to_latebench': datetime.now().isoformat() + 'Z',
                        'last_modified': datetime.now().isoformat() + 'Z',
                        'status': 'processed'
                    }
                }
                
                # Parse human annotation steps
                steps = []
                for i, step_data in enumerate(prm_data['label'].get('steps', [])):
                    if 'completions' not in step_data or not step_data['completions']:
                        continue
                        
                    completion = step_data['completions'][0]
                    step_text = completion.get('text', '').strip()
                    rating = completion.get('rating', 0)
                    
                    if not step_text:
                        continue
                    
                    is_error = (rating == -1)
                    importance = 'high' if is_error else ('medium' if rating == 1 else 'low')
                    
                    # Classify reasoning type
                    step_lower = step_text.lower()
                    if re.search(r'[+\-*/=]|calculate|compute', step_lower):
                        reasoning_type = 'calculation'
                    elif any(term in step_lower for term in ['solve', 'equation', 'substitute']):
                        reasoning_type = 'algebraic'
                    elif any(term in step_lower for term in ['therefore', 'thus', 'since', 'because']):
                        reasoning_type = 'logical'
                    else:
                        reasoning_type = 'analytical'
                    
                    steps.append({
                        'step_number': len(steps) + 1,
                        'content': step_text,
                        'importance': importance,
                        'reasoning_type': reasoning_type,
                        'is_error': is_error,
                        'is_modified': is_error  # Mark error steps as modified
                    })
                
                filtered_example['solution']['steps'] = steps
                filtered_examples.append(filtered_example)
                
            except Exception as e:
                print(f"   Error processing line {total_processed}: {e}")
                continue
    
    print(f"\n📊 Filtering Results:")
    print(f"   Total processed: {total_processed}")
    print(f"   Mapping failures: {mapping_failures}")
    print(f"   Level 5 problems found: {level5_found}")
    print(f"   Late error problems found: {late_error_found}")
    print(f"   Final filtered examples: {len(filtered_examples)}")
    
    return filtered_examples

def save_filtered_dataset(examples: List[Dict], output_file: str):
    """Save filtered examples to JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"✅ Saved {len(examples)} filtered examples to {output_file}")

def main():
    """Main function to run the filtering process"""
    print("🎯 PRM800K Level 5 Late Error Filter")
    print("=" * 50)
    
    # Filter examples
    filtered_examples = filter_prm800k_level5_late_errors(max_examples=200)
    
    if not filtered_examples:
        print("❌ No examples found matching the criteria")
        return
    
    # Save results
    output_file = './data/datasets/latebench_math_level5_natural_errors.json'
    save_filtered_dataset(filtered_examples, output_file)
    
    # Show sample statistics
    print(f"\n📈 Sample Analysis:")
    subjects = defaultdict(int)
    error_steps = []
    
    for example in filtered_examples:
        subjects[example['source']['subject']] += 1
        error_steps.append(example['source']['metadata']['first_error_step'])
    
    print(f"   Subject distribution:")
    for subject, count in sorted(subjects.items()):
        print(f"     {subject}: {count}")
    
    if error_steps:
        print(f"   Error step range: {min(error_steps)} - {max(error_steps)}")
        print(f"   Average error step: {sum(error_steps) / len(error_steps):.1f}")
    
    print(f"\n🎉 Successfully created filtered dataset with {len(filtered_examples)} examples!")
    print(f"📁 Dataset saved to: {output_file}")

if __name__ == "__main__":
    main()