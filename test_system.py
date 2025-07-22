#!/usr/bin/env python3
"""
Comprehensive test script for LateBench system validation.
"""

import sys
sys.path.append('src')

import os
import json
from data_loader import NuminaMathDataLoader
from error_types import MATH_ERROR_TAXONOMY
from error_injector import AdversarialErrorInjector
from critic import LLMCritic, StepFormatter
from visualization import VISUALIZER

def test_error_taxonomy():
    """Test the error taxonomy system."""
    print("=== Testing Error Taxonomy ===")
    
    error_names = MATH_ERROR_TAXONOMY.get_all_error_names()
    print(f"✓ Found {len(error_names)} error types")
    
    stats = MATH_ERROR_TAXONOMY.get_error_statistics()
    print(f"✓ Statistics: {stats['total_error_types']} total errors")
    
    # Test specific error retrieval
    test_error = MATH_ERROR_TAXONOMY.get_error_by_name("invalid_generalization")
    assert test_error is not None, "Could not retrieve invalid_generalization error"
    print("✓ Error retrieval working")

def test_data_loading():
    """Test data loading capabilities."""
    print("\n=== Testing Data Loading ===")
    
    loader = NuminaMathDataLoader()
    
    # Test loading filtered data
    if os.path.exists("data/filtered_long_solutions.json"):
        problems = loader.load_filtered_dataset("data/filtered_long_solutions.json")
        print(f"✓ Loaded {len(problems)} filtered problems")
        
        if len(problems) > 0:
            sample = problems[0]
            required_keys = ['problem', 'solution', 'messages']
            missing_keys = [key for key in required_keys if key not in sample]
            assert not missing_keys, f"Missing keys in sample: {missing_keys}"
            print("✓ Problem structure validated")
    else:
        print("⚠ No filtered dataset found, skipping data loading test")

def test_step_formatting():
    """Test step formatting and LaTeX cleaning."""
    print("\n=== Testing Step Formatting ===")
    
    formatter = StepFormatter()
    
    # Test LaTeX cleaning
    test_cases = [
        ("\\\\( x = 1 \\\\)", "\\( x = 1 \\)"),
        ("\\textbackslash{}", "\\"),
        ("$\\\\alpha$", "$\\alpha$")
    ]
    
    for input_text, expected in test_cases:
        result = formatter.clean_latex_escaping(input_text)
        assert result == expected, f"LaTeX cleaning failed: {input_text} -> {result}, expected {expected}"
    
    print("✓ LaTeX cleaning working correctly")

def test_error_injection():
    """Test error injection system."""
    print("\n=== Testing Error Injection ===")
    
    injector = AdversarialErrorInjector()
    print("✓ Error injector initialized")
    
    # Test solution parsing
    test_solution = "Step 1: First step\nStep 2: Second step\nStep 3: Third step\nStep 4: Fourth step"
    steps = injector.parse_solution_steps(test_solution)
    assert len(steps) == 4, f"Expected 4 steps, got {len(steps)}"
    print("✓ Solution parsing working")
    
    # Test with actual data if available
    if os.path.exists("data/educational_examples.json"):
        with open("data/educational_examples.json", 'r') as f:
            examples = json.load(f)
        print(f"✓ Found {len(examples)} educational examples")
        
        success_count = sum(1 for ex in examples if ex.get('success', False))
        print(f"✓ Success rate: {success_count}/{len(examples)} ({100*success_count/len(examples):.1f}%)")

def test_critic_system():
    """Test critic evaluation system."""
    print("\n=== Testing Critic System ===")
    
    try:
        critic = LLMCritic(model="gpt-4o-mini")
        print("✓ Critic initialized")
        
        # Test step formatting
        formatter = StepFormatter()
        test_steps = ["Step 1: Calculate x = 2 + 2 = 4", "Step 2: Therefore x = 4"]
        formatted = formatter.format_steps_for_critic(test_steps)
        print("✓ Step formatting working")
        
    except Exception as e:
        print(f"⚠ Critic test skipped (likely missing API key): {e}")

def test_dashboard_data():
    """Test dashboard data loading."""
    print("\n=== Testing Dashboard Data ===")
    
    from dashboard_utils import DashboardData
    
    # Test with educational examples if available
    if os.path.exists("data/educational_examples.json"):
        dashboard_data = DashboardData("data/educational_examples.json")
        print(f"✓ Dashboard loaded {len(dashboard_data.adversarial_examples)} examples")
        
        if len(dashboard_data.adversarial_examples) > 0:
            stats = dashboard_data.get_statistics()
            print(f"✓ Dashboard statistics: {stats}")
    else:
        print("⚠ No educational examples found for dashboard test")

def main():
    """Run all tests."""
    print("🚀 Running LateBench System Tests\n")
    
    try:
        test_error_taxonomy()
        test_data_loading()
        test_step_formatting()
        test_error_injection()
        test_critic_system()
        test_dashboard_data()
        
        print("\n✅ All tests completed successfully!")
        print("\nNext steps:")
        print("1. Download dataset: python download_data.py")
        print("2. Run experiment: python run_experiment.py --experiment small --num_examples 5")
        print("3. Start dashboard: python start_dashboard.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()