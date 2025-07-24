#!/usr/bin/env python3
"""
LateBench Test Suite Runner

Runs the organized test suite for LateBench system validation.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_test(test_file):
    """Run a single test file and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running: {test_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([
            sys.executable, 
            os.path.join('tests', test_file)
        ], capture_output=True, text=True, timeout=300)
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.1f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            return True
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            if result.stdout:
                print("Output:")
                print(result.stdout)
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False

def main():
    """Run all tests in the test suite."""
    
    print("🚀 LateBench Test Suite")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('tests'):
        print("❌ Error: tests/ directory not found. Run from project root.")
        sys.exit(1)
    
    # Get all test files
    test_files = []
    tests_dir = Path('tests')
    for test_file in tests_dir.glob('test_*.py'):
        test_files.append(test_file.name)
    
    if not test_files:
        print("❌ No test files found in tests/ directory")
        sys.exit(1)
    
    print(f"📋 Found {len(test_files)} test files:")
    for test_file in sorted(test_files):
        print(f"   - {test_file}")
    
    # Run tests
    print(f"\n🏃 Running tests...")
    start_time = time.time()
    
    passed = 0
    failed = 0
    
    for test_file in sorted(test_files):
        if run_test(test_file):
            passed += 1
        else:
            failed += 1
    
    total_time = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    if failed == 0:
        print(f"\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n💥 {failed} test(s) failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()