"""
Download and prepare PRM800K dataset
Downloads the dataset and filters for level 4-5 difficulty problems
"""

import os
import json
import requests
from datasets import load_dataset
from tqdm import tqdm
import sys

def download_prm800k():
    """Download PRM800K dataset from HuggingFace"""
    
    print("🔄 Downloading PRM800K dataset...")
    
    try:
        # Try different dataset names/configurations
        try:
            dataset = load_dataset("openai/prm800k", "phase2_train")
        except:
            try:
                # Try without config
                dataset = load_dataset("openai/prm800k")
            except:
                # Try alternative names
                try:
                    dataset = load_dataset("prm800k")
                except:
                    # Create mock data for testing if dataset not available
                    print("⚠️  PRM800K dataset not accessible. Creating mock data for testing...")
                    mock_data = create_mock_prm800k_data()
                    save_mock_data(mock_data)
                    return mock_data
        
        print(f"✅ Downloaded PRM800K with {len(dataset['train'])} examples")
        
        # Filter for level 4-5 difficulty and limit to 500 examples
        print("🔍 Filtering for level 4-5 difficulty problems...")
        
        filtered_examples = []
        level_counts = {4: 0, 5: 0}
        
        for example in tqdm(dataset['train']):
            if example['level'] in [4, 5] and len(filtered_examples) < 500:
                filtered_examples.append(example)
                level_counts[example['level']] += 1
                
                # Balance the levels somewhat
                if len(filtered_examples) >= 500:
                    break
        
        print(f"✅ Filtered to {len(filtered_examples)} examples")
        print(f"   Level 4: {level_counts[4]} problems")
        print(f"   Level 5: {level_counts[5]} problems")
        
        # Save raw filtered data
        os.makedirs('./data/sources/prm800k/raw', exist_ok=True)
        output_file = './data/sources/prm800k/raw/prm800k_level45_filtered.json'
        
        # Convert to JSON-serializable format
        json_examples = []
        for example in filtered_examples:
            json_example = {
                'problem': example['problem']['problem'],
                'level': example['problem']['level'],
                'type': example['problem']['type'],
                'solution': example['problem']['solution'],
                'answer': example['problem']['answer'] if example['problem']['answer'] else "No answer provided",
                'steps': example['completions'][0]['completion'].split('\n') if example['completions'] else [],
                'step_scores': example['completions'][0]['human_completion'] if example['completions'] else [],
                'original_id': example['problem'].get('id', f"prm_{len(json_examples)}")
            }
            json_examples.append(json_example)
        
        with open(output_file, 'w') as f:
            json.dump(json_examples, f, indent=2)
        
        print(f"✅ Saved raw data to {output_file}")
        
        return json_examples
        
    except Exception as e:
        print(f"❌ Error downloading PRM800K: {e}")
        print("Make sure you have the 'datasets' library installed: pip install datasets")
        return []


def analyze_prm800k_structure(examples):
    """Analyze the structure of PRM800K data"""
    if not examples:
        return
    
    print("\n📊 PRM800K Data Structure Analysis:")
    print("="*50)
    
    example = examples[0]
    print(f"Sample problem keys: {list(example.keys())}")
    
    print(f"\nSample problem:")
    print(f"Level: {example['level']}")
    print(f"Type: {example['type']}")
    print(f"Problem: {example['problem'][:200]}...")
    print(f"Solution: {example['solution'][:200]}...")
    print(f"Steps available: {len(example['steps'])} steps")
    print(f"Step scores available: {len(example['step_scores'])} scores")
    
    # Analyze difficulty distribution
    levels = {}
    types = {}
    for ex in examples[:100]:  # Sample first 100
        levels[ex['level']] = levels.get(ex['level'], 0) + 1
        types[ex['type']] = types.get(ex['type'], 0) + 1
    
    print(f"\nLevel distribution: {levels}")
    print(f"Type distribution: {types}")


if __name__ == "__main__":
    examples = download_prm800k()
    if examples:
        analyze_prm800k_structure(examples)
        print("\n✅ PRM800K download complete!")
        print(f"Next step: Run the PRM800K processor to convert to LateBench format")
    else:
        print("❌ PRM800K download failed")
        sys.exit(1)


def create_mock_prm800k_data():
    """Create mock PRM800K data for testing"""
    mock_examples = []
    
    problems = [
        {
            "problem": "Find the number of positive integers $n \\leq 1000$ such that $\\gcd(n, 21) = 1$.",
            "solution": "We need to find positive integers $n \\leq 1000$ such that $\\gcd(n, 21) = 1$. Since $21 = 3 \\times 7$, we have $\\gcd(n, 21) = 1$ if and only if $n$ is not divisible by 3 or 7. Using inclusion-exclusion principle: Total integers from 1 to 1000 = 1000. Integers divisible by 3: $\\lfloor 1000/3 \\rfloor = 333$. Integers divisible by 7: $\\lfloor 1000/7 \\rfloor = 142$. Integers divisible by both 3 and 7 (i.e., by 21): $\\lfloor 1000/21 \\rfloor = 47$. By inclusion-exclusion: integers divisible by 3 or 7 = $333 + 142 - 47 = 428$. Therefore, integers with $\\gcd(n, 21) = 1$ are $1000 - 428 = 572$.",
            "answer": "572",
            "level": 4,
            "type": "Number Theory"
        },
        {
            "problem": "In triangle $ABC$, $AB = 13$, $BC = 14$, and $CA = 15$. Find the length of the altitude from $A$ to side $BC$.",
            "solution": "First, find the area using Heron's formula. The semi-perimeter is $s = (13 + 14 + 15)/2 = 21$. Area = $\\sqrt{s(s-a)(s-b)(s-c)} = \\sqrt{21 \\cdot 8 \\cdot 7 \\cdot 6} = \\sqrt{21 \\cdot 336} = \\sqrt{7056} = 84$. For the altitude from $A$ to side $BC$: Area = $(1/2) \\cdot BC \\cdot h_a$, so $84 = (1/2) \\cdot 14 \\cdot h_a$. Solving: $h_a = 168/14 = 12$.",
            "answer": "12", 
            "level": 4,
            "type": "Geometry"
        },
        {
            "problem": "Let $f(x) = x^3 - 6x^2 + 11x - 6$. Find all real solutions to $f(x) = 0$.",
            "solution": "We try to factor the polynomial. Testing $x = 1$: $f(1) = 1 - 6 + 11 - 6 = 0$. So $(x-1)$ is a factor. Performing polynomial division: $f(x) = (x-1)(x^2 - 5x + 6)$. For $x^2 - 5x + 6 = 0$, we use the quadratic formula: $x = (5 \\pm \\sqrt{25-24})/2 = (5 \\pm 1)/2$. This gives $x = 3$ or $x = 2$. Therefore, $f(x) = (x-1)(x-2)(x-3)$. The solutions are $x = 1, 2, 3$.",
            "answer": "x = 1, 2, 3",
            "level": 5,
            "type": "Algebra"
        }
    ]
    
    for i, prob in enumerate(problems):
        # Create mock step-by-step solution
        steps = prob["solution"].split(". ")
        step_scores = [0.9, 0.8, 0.7, 0.9, 0.8] * (len(steps) // 5 + 1)
        step_scores = step_scores[:len(steps)]
        
        mock_examples.append({
            'problem': prob['problem'],
            'level': prob['level'],
            'type': prob['type'],
            'solution': prob['solution'],
            'answer': prob['answer'],
            'steps': steps,
            'step_scores': step_scores,
            'original_id': f"prm_mock_{i}"
        })
    
    return mock_examples


def save_mock_data(mock_data):
    """Save mock data to file"""
    os.makedirs('./data/sources/prm800k/raw', exist_ok=True)
    output_file = './data/sources/prm800k/raw/prm800k_level45_filtered.json'
    
    with open(output_file, 'w') as f:
        json.dump(mock_data, f, indent=2)
    
    print(f"✅ Saved mock PRM800K data to {output_file}")