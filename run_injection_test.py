#!/usr/bin/env python3
"""
Test error injection and show UI instructions.
"""

import requests
import json

def run_injection_test():
    """Run injection test and show UI instructions."""
    
    base_url = "http://localhost:8000"
    example_id = "lb_numinamath_c16346d7"  # From debug output
    
    print("🛠️  ERROR INJECTION TROUBLESHOOTING")
    print("="*50)
    
    print("\n✅ STATUS CHECK:")
    print(f"   Dashboard: Running at {base_url}")
    print(f"   Current dataset: numinamath (all)")
    print(f"   Example to use: {example_id}")
    print(f"   Custom suggestions: 5 available")
    print(f"   Solution steps: 8 (sufficient for injection)")
    
    print("\n🚀 TESTING API INJECTION...")
    try:
        response = requests.post(
            f"{base_url}/api/manual_injection/{example_id}",
            json={'user_remarks': 'Testing if injection works'},
            headers={'Content-Type': 'application/json'},
            timeout=45
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ API INJECTION WORKS!")
                injection_result = result['injection_result']
                error_step = injection_result.get('error_analysis', {}).get('selected_error_step')
                error_type = injection_result.get('error_analysis', {}).get('error_type')
                print(f"   Error injected in step: {error_step}")
                print(f"   Error type: {error_type}")
            else:
                print(f"❌ API injection failed: {result.get('error')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            if response.status_code == 500:
                try:
                    error_data = response.json()
                    print(f"   Server error: {error_data.get('error', 'Unknown')}")
                except:
                    pass
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n" + "="*60)
    print("📋 HOW TO RUN ERROR INJECTION IN THE UI")
    print("="*60)
    
    print("\n🌐 1. OPEN BROWSER:")
    print("   Go to: http://localhost:8000")
    print("   You should see a 5-panel dashboard")
    
    print("\n📍 2. LOCATE THE RIGHT PANEL:")
    print("   Look for 'Manual Error Injection' on the RIGHT side")
    print("   This panel has 3 sections:")
    print("   - Error Suggestion (top)")
    print("   - Injection Attempt (middle)") 
    print("   - Final Decision (bottom)")
    
    print("\n📝 3. STEP-BY-STEP PROCESS:")
    
    print("\n   STEP A: Add Error Suggestion (if needed)")
    print("   - Find textarea: 'Describe the type of error you want to inject...'")
    print("   - Type: 'Add calculation error in final step'")
    print("   - Click: 'Save Suggestion' button")
    print("   - Watch: Button changes to 'Saving...' → '✅ Saved'")
    print("   - Verify: Suggestion appears in list below")
    
    print("\n   STEP B: Run Error Injection")
    print("   - Find: 'Inject Error' button (orange/blue button)")
    print("   - Optionally add remarks in textarea above it")
    print("   - Click: 'Inject Error' button")
    print("   - Watch: Progress panel appears with 4 animated steps:")
    print("     🔄 Validating prerequisites...")
    print("     🔄 Sending request to GPT-4...")
    print("     🔄 Processing AI response...")
    print("     🔄 Finalizing injection...")
    print("   - Wait: 15-30 seconds for GPT-4 to process")
    print("   - Success: Green checkmarks ✅ appear")
    
    print("\n   STEP C: Review Results")
    print("   - Check: Modified solution panel (center) for error highlights")
    print("   - Review: Injection attempt appears in attempts list")
    print("   - Decide: Click 'Yes', 'Maybe', or 'No' button")
    
    print("\n🔧 COMMON ISSUES & SOLUTIONS:")
    print("   ❌ 'Nothing happens when I click Inject Error'")
    print("      → Check browser console (F12) for JavaScript errors")
    print("      → Make sure you're clicking the right button")
    print("      → Try refreshing the page")
    
    print("\n   ❌ 'Button shows loading but nothing happens'")
    print("      → Wait longer (GPT-4 can take 30+ seconds)")
    print("      → Check internet connection")
    print("      → Verify OPENAI_API_KEY is set")
    
    print("\n   ❌ 'Error: Solution too short' or 'No custom suggestions'")
    print("      → Add custom suggestion first")
    print("      → Switch to NuminaMath dataset (has longer solutions)")
    
    print("\n   ❌ 'Progress panel doesn't appear'")
    print("      → JavaScript may be disabled")
    print("      → Try different browser")
    print("      → Check browser console for errors")
    
    print("\n🎯 QUICK TEST:")
    print("   1. Open http://localhost:8000")
    print("   2. Find the rightmost panel 'Manual Error Injection'")
    print("   3. Click the 'Inject Error' button")
    print("   4. Watch for progress panel to appear")
    print("   5. Wait 15-30 seconds for completion")
    
    print("\n💡 WHAT YOU SHOULD SEE:")
    print("   - Progress panel with animated steps")
    print("   - Real-time status updates")
    print("   - Success message with error details")
    print("   - Modified solution with highlighted errors")
    print("   - New attempt in attempts list")
    
    print("\n" + "="*60)
    print("🚀 Your example is ready: lb_numinamath_c16346d7")
    print("💻 Dashboard URL: http://localhost:8000")
    print("📞 If still having issues, check browser console (F12)")

if __name__ == "__main__":
    run_injection_test()