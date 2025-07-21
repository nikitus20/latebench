#!/usr/bin/env python3
"""
Start the LateBench dashboard.
"""

import sys
import os
sys.path.append('src')

from src.dashboard import app, initialize_data

if __name__ == "__main__":
    print("🚀 Starting LateBench Dashboard...")
    
    # Initialize data
    initialize_data()
    
    # Start Flask app
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("⌨️  Keyboard shortcuts:")
    print("   - Left/Right arrows: Navigate between examples")
    print("   - Ctrl+R: Run critic evaluation")
    print("   - Ctrl+E: Export current example")
    print("\n✨ Features:")
    print("   - Browse adversarial examples with filtering")
    print("   - Compare original vs modified solutions")
    print("   - Run GPT-4o-mini critic evaluations")
    print("   - Analyze critic performance vs ground truth")
    print("   - Export examples as JSON")
    
    app.run(debug=False, host='0.0.0.0', port=8000)