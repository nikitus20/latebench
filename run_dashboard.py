#!/usr/bin/env python3
"""
LateBench Dashboard Entry Point
"""

import os
import sys
import logging
from datetime import datetime

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dashboard.app import app, initialize_data

def setup_logging():
    """Configure logging for the dashboard."""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/dashboard_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Dashboard logging initialized. Log file: {log_file}")

def main():
    """Main entry point for the dashboard."""
    setup_logging()
    
    # Initialize data
    initialize_data()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '127.0.0.1')  # Changed from 0.0.0.0 to localhost
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting LateBench Dashboard on {host}:{port}")
    print(f"Starting LateBench Dashboard...")
    print(f"Open http://localhost:{port} in your browser")
    
    app.run(host=host, port=port, debug=debug)

if __name__ == '__main__':
    main()