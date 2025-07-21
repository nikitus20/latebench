#!/usr/bin/env python3
"""
Test script for the LLM critic functionality.
"""

import sys
sys.path.append('src')

import json
from critic import LLMCritic, evaluate_single_example
from dashboard_utils import DashboardData, analyze_critic_performance

def test_step_formatting():
    """Test step formatting and cleaning utilities."""
    print("=== Testing Step Formatting ===")
    
    from critic import StepFormatter
    
    # Test LaTeX cleaning
    test_cases = [
        "\\\\($\\\\sin C = \\\\frac{1}{2}$\\\\)",
        "\\textbackslash{}cos C = \\frac{1}{3}",
        "Step 1: Using the sine law, we have..."
    ]
    
    for i, case in enumerate(test_cases, 1):
        cleaned = StepFormatter.clean_latex_escaping(case)
        print(f"Test {i}:")
        print(f"  Input:  {case}")
        print(f"  Output: {cleaned}")
        print()
    
    print("✓ Step formatting tests complete\n")

def test_critic_initialization():
    """Test critic initialization."""
    print("=== Testing Critic Initialization ===")
    
    try:
        critic = LLMCritic(model="gpt-4o-mini")
        print("✓ Critic initialized successfully")
        print(f"  Model: {critic.model}")
        print(f"  Has system prompt: {bool(critic.system_prompt)}")
        print(f"  Has user prompt template: {bool(critic.user_prompt_template)}")
        return critic
    except Exception as e:
        print(f"✗ Error initializing critic: {e}")
        return None

def test_data_loading():
    """Test loading adversarial examples."""
    print("=== Testing Data Loading ===")
    
    try:
        data = DashboardData("./data/small_experiment_results.json")
        print(f"✓ Loaded {len(data.examples)} examples")
        
        if data.examples:
            example = data.examples[0]
            print(f"  First example ID: {example['id']}")
            print(f"  Title: {example['title']}")
            print(f"  Error type: {example['error_info']['type']}")
            print(f"  Error step: {example['error_info']['step']}")
            print(f"  Number of original steps: {example['original_solution']['num_steps']}")
            print(f"  Number of modified steps: {example['modified_solution']['num_steps']}")
            return data
        else:
            print("  No examples found")
            return None
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def test_single_critic_evaluation(critic, data):
    """Test critic evaluation on a single example."""
    print("=== Testing Single Critic Evaluation ===")
    
    if not critic or not data or not data.examples:
        print("✗ Cannot run test - missing critic or data")
        return None
    
    try:
        example = data.examples[0]
        print(f"Evaluating example: {example['title']}")
        print(f"Ground truth error at step: {example['error_info']['step']}")
        print(f"Error type: {example['error_info']['type']}")
        
        # Create raw example for critic
        raw_example = {
            'original_problem': {
                'problem': example['problem'],
                'parsed_steps': [step['content'] for step in example['original_solution']['steps']]
            },
            'modified_solution': {
                'steps': [
                    {
                        'step_num': step['number'],
                        'content': step['content'],
                        'modified': step['is_modified'],
                        'error': step['is_error']
                    }
                    for step in example['modified_solution']['steps']
                ]
            }
        }
        
        print("\nRunning critic evaluation...")
        result = evaluate_single_example(raw_example, model="gpt-4o-mini")
        
        print(f"\n✓ Critic evaluation complete")
        print(f"  Processing time: {result.processing_time:.2f}s")
        print(f"  Has errors: {result.has_errors}")
        print(f"  Error steps detected: {result.error_steps}")
        
        if result.has_errors:
            for step_num in result.error_steps:
                explanation = result.explanations.get(step_num, "No explanation")
                print(f"  Step {step_num}: {explanation[:100]}...")
        
        # Analyze performance
        data.add_critic_result(example['id'], result)
        analysis = analyze_critic_performance(data.get_example(example['id']))
        
        print(f"\n📊 Performance Analysis:")
        print(f"  Correct detection: {analysis['correct_detection']}")
        print(f"  Exact step match: {analysis['exact_step_match']}")
        print(f"  False positives: {analysis['false_positives']}")
        print(f"  Missed error: {analysis['missed_error']}")
        
        return result
        
    except Exception as e:
        print(f"✗ Error running critic evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_batch_evaluation(critic, data, max_examples=2):
    """Test batch evaluation on multiple examples."""
    print(f"=== Testing Batch Evaluation ({max_examples} examples) ===")
    
    if not critic or not data or not data.examples:
        print("✗ Cannot run test - missing critic or data")
        return None
    
    try:
        # Select examples for testing
        examples_to_test = data.examples[:max_examples]
        
        print(f"Running batch evaluation on {len(examples_to_test)} examples...")
        
        results = []
        for i, example in enumerate(examples_to_test):
            print(f"\nEvaluating example {i+1}/{len(examples_to_test)}: {example['title']}")
            
            # Create raw example
            raw_example = {
                'original_problem': {
                    'problem': example['problem'],
                    'parsed_steps': [step['content'] for step in example['original_solution']['steps']]
                },
                'modified_solution': {
                    'steps': [
                        {
                            'step_num': step['number'],
                            'content': step['content'],
                            'modified': step['is_modified'],
                            'error': step['is_error']
                        }
                        for step in example['modified_solution']['steps']
                    ]
                }
            }
            
            # Evaluate
            result = evaluate_single_example(raw_example, model="gpt-4o-mini")
            results.append(result)
            
            # Store result
            data.add_critic_result(example['id'], result)
            
            # Quick feedback
            if result.has_errors:
                print(f"  ✓ Found {len(result.error_steps)} errors in steps: {result.error_steps}")
            else:
                print(f"  - No errors detected")
        
        # Summary analysis
        print(f"\n📊 Batch Evaluation Summary:")
        successful = sum(1 for r in results if r.has_errors)
        print(f"  Examples with detected errors: {successful}/{len(results)}")
        
        correct_detections = 0
        exact_matches = 0
        
        for i, result in enumerate(results):
            example = examples_to_test[i]
            data.add_critic_result(example['id'], result)
            analysis = analyze_critic_performance(data.get_example(example['id']))
            
            if analysis['correct_detection']:
                correct_detections += 1
            if analysis['exact_step_match']:
                exact_matches += 1
        
        print(f"  Correct error detections: {correct_detections}/{len(results)}")
        print(f"  Exact step matches: {exact_matches}/{len(results)}")
        
        # Save results
        data.save_critic_results()
        print(f"  Results saved to dashboard")
        
        return results
        
    except Exception as e:
        print(f"✗ Error in batch evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run all tests."""
    print("🧪 LateBench Critic Testing Suite\n")
    
    # Test 1: Step formatting
    test_step_formatting()
    
    # Test 2: Critic initialization
    critic = test_critic_initialization()
    if not critic:
        print("❌ Cannot continue without working critic")
        return 1
    
    print()
    
    # Test 3: Data loading
    data = test_data_loading()
    if not data or not data.examples:
        print("❌ Cannot continue without test data")
        print("Please run: python run_experiment.py --experiment small --num_examples 2")
        return 1
    
    print()
    
    # Test 4: Single evaluation
    single_result = test_single_critic_evaluation(critic, data)
    if not single_result:
        print("❌ Single evaluation failed")
        return 1
    
    print()
    
    # Test 5: Batch evaluation (if user wants it)
    print("Would you like to run batch evaluation? This will use more API credits.")
    response = input("Run batch evaluation? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        batch_results = test_batch_evaluation(critic, data, max_examples=2)
        if batch_results:
            print("\n✅ All tests completed successfully!")
            print("\n🚀 You can now start the dashboard:")
            print("  python src/dashboard.py")
        else:
            print("❌ Batch evaluation failed")
            return 1
    else:
        print("\n✅ Core tests completed successfully!")
        print("\n🚀 You can now start the dashboard:")
        print("  python src/dashboard.py")
    
    return 0

if __name__ == "__main__":
    exit(main())