#!/usr/bin/env python3
"""
Launch the simplified LateBench Dashboard.
Core functionality: manual error injection review and curation workflow.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import dashboard app
sys.path.insert(0, str(Path(__file__).parent.parent / "dashboard"))
from simple_app import app

import argparse


def main():
    parser = argparse.ArgumentParser(description='Launch LateBench Dashboard')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    print("🚀 LateBench Simplified Dashboard")
    print("=" * 50)
    print(f"🌐 Server: http://{args.host}:{args.port}")
    print("📊 Features:")
    print("   • Dataset loading and navigation")
    print("   • Interactive error injection with custom suggestions")
    print("   • Critic evaluation integration")
    print("   • Manual review and decision tracking")
    print("   • Progress monitoring and export")
    print("=" * 50)
    print("🔧 Usage:")
    print("   1. Select and load a dataset")
    print("   2. Navigate through examples")
    print("   3. Inject errors (with optional custom suggestions)")
    print("   4. Review results and make decisions (Yes/Maybe/No)")
    print("   5. Export your decisions for analysis")
    print("=" * 50)
    
    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard shutdown complete!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())