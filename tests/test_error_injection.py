#!/usr/bin/env python3
"""
Test script to verify error injection workflow functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import json
from dashboard.utils import DashboardData

def test_error_injection_workflow():
    """Test the complete error injection workflow."""
    
    # Initialize dashboard data to get an example
    dashboard_data = DashboardData()
    dashboard_data.switch_dataset('numinamath', 'all')
    
    if not dashboard_data.examples:
        print("❌ No examples loaded")
        return False
    
    # Get first example
    example = dashboard_data.examples[0]
    example_id = example['id']
    
    print(f"🎯 Testing error injection with example: {example_id}")
    print(f"Problem: {example['problem'][:100]}..." if isinstance(example['problem'], str) else f"Problem: {example.get('title', 'Unknown problem')}")
    
    # Test 1: Save a custom suggestion
    print("\n📝 Step 1: Saving custom error suggestion...")
    suggestion_data = {
        'suggestion': 'Add an algebraic error in step 5 where we incorrectly simplify the cosine calculation, leading to a wrong final area value.'
    }
    
    try:
        # Simulate saving suggestion (would normally be via API)
        dashboard_data.update_custom_suggestion(example_id, suggestion_data['suggestion'])
        dashboard_data.save_manual_injection_data()
        print("✅ Custom suggestion saved successfully")
    except Exception as e:
        print(f"❌ Error saving suggestion: {e}")
        return False
    
    # Test 2: Check manual injection data
    print("\n📊 Step 2: Verifying manual injection data...")
    try:
        manual_data = dashboard_data.get_manual_injection_data(example_id)
        if manual_data['custom_suggestions']:
            print(f"✅ Found {len(manual_data['custom_suggestions'])} custom suggestions")
            print(f"Latest suggestion: {manual_data['custom_suggestions'][-1][:50]}...")
        else:
            print("❌ No custom suggestions found")
            return False
    except Exception as e:
        print(f"❌ Error retrieving manual data: {e}")
        return False
    
    # Test 3: Prepare injection (simulate the API call structure)
    print("\n🔧 Step 3: Testing error injection preparation...")
    try:
        from src.error_injector import AdversarialErrorInjector
        
        # Create injector instance
        injector = AdversarialErrorInjector()
        
        # Debug: Check example structure
        print(f"Example keys: {list(example.keys())}")
        print(f"Original steps available: {'original_steps' in example}")
        print(f"Modified solution steps: {len(example.get('modified_solution', {}).get('steps', []))}")
        
        # Prepare the original problem data for injection using the correct structure
        modified_steps = example.get('modified_solution', {}).get('steps', [])
        raw_example = {
            'problem': example['problem'] if isinstance(example['problem'], str) else example.get('title', ''),
            'solution': '\n'.join([step['content'] for step in modified_steps]),
            'answer': example.get('modified_solution', {}).get('final_answer', 'No answer')
        }
        
        print("✅ Error injector initialized successfully")
        print(f"Using modified solution steps: {len(modified_steps)} steps")
        
        # Test injection with custom suggestion
        custom_suggestion = manual_data['custom_suggestions'][-1]
        print(f"Using custom suggestion: {custom_suggestion[:80]}...")
        
        # This would normally be called via the API
        print("\n🚀 Step 4: Running error injection...")
        injection_result = injector.inject_error_with_custom_suggestion(
            raw_example,
            custom_suggestion=custom_suggestion,
            max_retries=1  # Reduced for testing
        )
        
        if injection_result and injection_result.success:
            print("✅ Error injection successful!")
            print(f"Modified solution has {len(injection_result.modified_solution.get('steps', []))} steps")
            print(f"Error type: {injection_result.error_analysis.get('error_type', 'Unknown')}")
            print(f"Error explanation: {injection_result.error_explanation[:100]}...")
            
            # Test 5: Simulate saving the attempt
            attempt_data = {
                'user_remarks': 'Testing error injection workflow',
                'injection_result': injection_result.__dict__,
                'custom_suggestion': custom_suggestion
            }
            
            dashboard_data.add_injection_attempt(example_id, attempt_data)
            dashboard_data.save_manual_injection_data()
            print("✅ Injection attempt saved successfully")
            
            return True
        else:
            error_msg = injection_result.error_message if injection_result else "Unknown error"
            print(f"❌ Error injection failed: {error_msg}")
            return False
            
    except Exception as e:
        print(f"❌ Error during injection: {e}")
        return False

def main():
    """Run the test."""
    print("=== LateBench Error Injection Workflow Test ===\n")
    
    # Check prerequisites
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ No OPENAI_API_KEY found in environment")
        print("Skipping API-dependent tests...")
        return
    
    success = test_error_injection_workflow()
    
    if success:
        print("\n🎉 All tests passed! Error injection workflow is functional.")
        print("\nNext steps:")
        print("1. Open dashboard at http://localhost:8000")
        print("2. Navigate to any NuminaMath example")
        print("3. Use the manual error injection interface")
        print("4. Test the complete workflow: suggestion → injection → decision")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")

if __name__ == "__main__":
    main()