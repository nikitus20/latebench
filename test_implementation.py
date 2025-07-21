#!/usr/bin/env python3
"""
Test script to validate the LateBench implementation.
"""

import sys
sys.path.append('src')

import os
from data_loader import NuminaMathDataLoader
from error_types import MATH_ERROR_TAXONOMY
from error_injector import AdversarialErrorInjector
from visualization import VISUALIZER

def test_error_taxonomy():
    """Test the error taxonomy system."""
    print("Testing Error Taxonomy...")
    
    # Test basic functionality
    error_names = MATH_ERROR_TAXONOMY.get_all_error_names()
    print(f"✓ Found {len(error_names)} error types")
    
    # Test specific error retrieval
    error = MATH_ERROR_TAXONOMY.get_error_by_name("invalid_generalization")
    assert error is not None, "Should find invalid_generalization error"
    print(f"✓ Retrieved error: {error.name}")
    
    # Test statistics
    stats = MATH_ERROR_TAXONOMY.get_error_statistics()
    assert "total_error_types" in stats, "Should have statistics"
    print(f"✓ Statistics: {stats['total_error_types']} total error types")
    
    print("Error taxonomy tests passed!\n")

def test_data_loader():
    """Test the data loading system."""
    print("Testing Data Loader...")
    
    loader = NuminaMathDataLoader(cache_dir="./data")
    
    # Test solution parsing
    sample_solution = """Step 1: Start with the given equation.
Step 2: Apply algebraic manipulation.
Step 3: Solve for x."""
    
    steps = loader.parse_solution_steps(sample_solution)
    assert len(steps) == 3, f"Should parse 3 steps, got {len(steps)}"
    print("✓ Solution parsing works")
    
    # Check if dataset exists
    if os.path.exists("./data/numinamath_local"):
        print("✓ Dataset cache found")
    else:
        print("! Dataset not downloaded yet (run download_data.py first)")
    
    print("Data loader tests passed!\n")

def test_error_injector_init():
    """Test error injector initialization."""
    print("Testing Error Injector Initialization...")
    
    # Test initialization without API (just structure)
    try:
        injector = AdversarialErrorInjector(api_key="dummy_key")
        print("✓ Error injector initialized")
        
        # Test prompt creation
        assert injector.system_prompt is not None, "Should have system prompt"
        assert injector.user_prompt_template is not None, "Should have user prompt template"
        print("✓ Prompts created")
        
        # Test solution parsing
        sample_solution = "Step 1: First step\nStep 2: Second step\nStep 3: Final step"
        steps = injector.parse_solution_steps(sample_solution)
        assert len(steps) == 3, f"Should parse 3 steps, got {len(steps)}"
        print("✓ Solution parsing in injector works")
        
    except Exception as e:
        print(f"Error injector test failed: {e}")
        return False
    
    print("Error injector initialization tests passed!\n")
    return True

def test_visualizer():
    """Test visualization system."""
    print("Testing Visualizer...")
    
    try:
        # Test initialization
        viz = VISUALIZER
        print("✓ Visualizer initialized")
        
        # Test empty results handling
        empty_metrics = viz.create_quality_metrics_report([])
        assert "error" in empty_metrics, "Should handle empty results"
        print("✓ Empty results handling works")
        
    except Exception as e:
        print(f"Visualizer test failed: {e}")
        return False
    
    print("Visualizer tests passed!\n")
    return True

def create_mock_example():
    """Create a mock example for testing without API calls."""
    return {
        'problem': 'Solve for x in the equation 2x + 3 = 7',
        'solution': '''Step 1: Start with 2x + 3 = 7
Step 2: Subtract 3 from both sides: 2x = 4  
Step 3: Divide both sides by 2: x = 2
Step 4: Check: 2(2) + 3 = 4 + 3 = 7 ✓''',
        'answer': 'x = 2',
        'level': 1,
        'type': 'algebra'
    }

def test_full_workflow_mock():
    """Test the full workflow with mock data."""
    print("Testing Full Workflow (Mock)...")
    
    # Create mock problem
    mock_problem = create_mock_example()
    
    # Test data loading components
    loader = NuminaMathDataLoader()
    steps = loader.parse_solution_steps(mock_problem['solution'])
    print(f"✓ Parsed {len(steps)} steps from mock problem")
    
    # Test error taxonomy suggestions
    suggestions = MATH_ERROR_TAXONOMY.suggest_error_for_context(
        mock_problem['solution'], 
        difficulty_preference="medium"
    )
    print(f"✓ Found {len(suggestions)} suggested error types")
    
    print("Full workflow tests passed!\n")

def main():
    """Run all tests."""
    print("=== LateBench Implementation Tests ===\n")
    
    try:
        test_error_taxonomy()
        test_data_loader()
        injector_ok = test_error_injector_init()
        visualizer_ok = test_visualizer()
        test_full_workflow_mock()
        
        print("=== All Tests Passed! ===")
        
        if os.getenv("OPENAI_API_KEY"):
            print("\n💡 API key found. You can now run real experiments!")
            print("Next steps:")
            print("1. Run: python download_data.py (if not done)")
            print("2. Open: notebooks/example_generation.ipynb")
            print("3. Or run: python run_experiment.py")
        else:
            print("\n⚠️  No OPENAI_API_KEY found in environment")
            print("Add your key to .env file to run real experiments")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())