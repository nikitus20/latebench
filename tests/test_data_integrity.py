#!/usr/bin/env python3
"""
Test script to generate separated PRM800K datasets for complete solutions vs error-labeled problems
"""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.prm800k_processor import PRM800KProcessor

def main():
    # Initialize processor
    processor = PRM800KProcessor()
    
    # File paths
    input_file = "./data/sources/prm800k/prm800k/data/phase2_train.jsonl"
    output_file = "./data/datasets/latebench_prm800k_raw.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        print("Please run download_prm800k.py first to download the dataset")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print("Processing PRM800K dataset with separation by error status...")
    print(f"Input: {input_file}")
    print(f"Output base: {output_file}")
    
    # Process with separation enabled (limit to 100 examples for testing)
    processed_examples = processor.process_dataset(
        input_file=input_file,
        output_file=output_file,
        max_examples=100,  # Small test sample
        separate_by_errors=True
    )
    
    print(f"\nProcessing completed!")
    print(f"Total examples processed: {len(processed_examples)}")
    
    # Check if separated files were created
    complete_file = output_file.replace('.json', '_complete.json')
    error_file = output_file.replace('.json', '_errors.json')
    
    if os.path.exists(complete_file):
        print(f"✅ Complete solutions file created: {complete_file}")
    if os.path.exists(error_file):
        print(f"✅ Error-labeled problems file created: {error_file}")
    
    print(f"✅ Combined dataset file created: {output_file}")

if __name__ == "__main__":
    main()