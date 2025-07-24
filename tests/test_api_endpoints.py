#!/usr/bin/env python3
"""
Test the error injection API workflow directly.
"""

import requests
import json
import time

def test_api_workflow():
    """Test the complete API workflow."""
    
    base_url = "http://localhost:8000"
    
    print("=== Testing LateBench Error Injection API Workflow ===\n")
    
    # Step 0: Switch to NuminaMath dataset (has longer examples)
    print("🔄 Step 0: Switching to NuminaMath dataset...")
    try:
        response = requests.post(
            f"{base_url}/api/switch_dataset",
            json={'dataset_name': 'numinamath', 'problem_type': 'all'},
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("✅ Switched to NuminaMath dataset")
        else:
            print(f"⚠️ Failed to switch dataset: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error switching dataset: {e}")
    
    # Step 1: Get available examples
    print("\n📚 Step 1: Getting available examples...")
    try:
        response = requests.get(f"{base_url}/api/examples")
        if response.status_code == 200:
            data = response.json()
            examples = data.get('examples', [])
            if examples:
                # Find an example with sufficient steps
                suitable_example = None
                for ex in examples:
                    steps = ex.get('modified_solution', {}).get('steps', [])
                    if len(steps) >= 4:
                        suitable_example = ex
                        break
                
                if suitable_example:
                    example_id = suitable_example['id']
                    steps_count = len(suitable_example.get('modified_solution', {}).get('steps', []))
                    print(f"✅ Found {len(examples)} examples")
                    print(f"Using example: {example_id} ({steps_count} steps)")
                else:
                    example_id = examples[0]['id']
                    print(f"✅ Found {len(examples)} examples")
                    print(f"Using example: {example_id} (may be too short)")
            else:
                print("❌ No examples found")
                return False
        else:
            print(f"❌ Failed to get examples: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting examples: {e}")
        return False
    
    # Step 2: Save a custom suggestion
    print(f"\n📝 Step 2: Saving custom error suggestion for {example_id}...")
    suggestion_data = {
        'suggestion': 'Introduce a calculation error in the final steps where we incorrectly compute the area, leading to a wrong final answer.'
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/save_suggestion/{example_id}",
            json=suggestion_data,
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code == 200:
            print("✅ Custom suggestion saved successfully")
        else:
            print(f"❌ Failed to save suggestion: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error saving suggestion: {e}")
        return False
    
    # Step 3: Get manual injection data
    print(f"\n📊 Step 3: Checking manual injection data...")
    try:
        response = requests.get(f"{base_url}/api/manual_data/{example_id}")
        if response.status_code == 200:
            data = response.json()
            if data['success'] and data['data']['custom_suggestions']:
                print(f"✅ Found {len(data['data']['custom_suggestions'])} custom suggestions")
            else:
                print("❌ No custom suggestions found")
                return False
        else:
            print(f"❌ Failed to get manual data: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting manual data: {e}")
        return False
    
    # Step 4: Run manual error injection
    print(f"\n🚀 Step 4: Running manual error injection...")
    injection_data = {
        'user_remarks': 'Testing API workflow'
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/manual_injection/{example_id}",
            json=injection_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print("✅ Error injection successful!")
                result = data['injection_result']
                print(f"   Attempt number: {result.get('attempt_number', 1)}")
                error_explanation = result.get('error_explanation', {})
                if isinstance(error_explanation, dict):
                    what_changed = error_explanation.get('what_changed', '')[:100]
                else:
                    what_changed = str(error_explanation)[:100]
                print(f"   Error explanation: {what_changed}...")
                if result.get('modified_solution'):
                    print(f"   Modified solution created successfully")
            else:
                print(f"❌ Error injection failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Failed to run injection: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error running injection: {e}")
        return False
    
    # Step 5: Set final decision
    print(f"\n✅ Step 5: Setting final decision...")
    decision_data = {
        'decision': 'yes'
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/set_decision/{example_id}",
            json=decision_data,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                print(f"✅ Decision set to: {data['decision']}")
            else:
                print(f"❌ Failed to set decision: {data.get('error')}")
                return False
        else:
            print(f"❌ Failed to set decision: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error setting decision: {e}")
        return False
    
    # Step 6: Get injection history
    print(f"\n📜 Step 6: Getting injection history...")
    try:
        response = requests.get(f"{base_url}/api/injection_history/{example_id}")
        if response.status_code == 200:
            data = response.json()
            if data['success']:
                history = data['history']
                suggestions = data['suggestions']
                decision = data['final_decision']
                print(f"✅ History retrieved:")
                print(f"   Injection attempts: {len(history)}")
                print(f"   Custom suggestions: {len(suggestions)}")
                print(f"   Final decision: {decision}")
            else:
                print(f"❌ Failed to get history: {data.get('error')}")
                return False
        else:
            print(f"❌ Failed to get history: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting history: {e}")
        return False
    
    return True

def main():
    """Main test function."""
    
    # Wait a moment for dashboard to be ready
    print("Waiting for dashboard to be ready...")
    time.sleep(2)
    
    success = test_api_workflow()
    
    if success:
        print("\n🎉 Complete API workflow test passed!")
        print("\nThe error injection system is fully functional:")
        print("✅ Custom suggestions can be saved")
        print("✅ Manual error injection works")
        print("✅ Final decisions can be recorded")
        print("✅ Injection history is maintained")
        print("\nYou can now use the dashboard interface at http://localhost:8000")
    else:
        print("\n❌ API workflow test failed - check dashboard is running on port 8000")

if __name__ == "__main__":
    main()