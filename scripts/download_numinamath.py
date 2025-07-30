#!/usr/bin/env python3
"""
Download NuminaMath dataset from Hugging Face
Simple and clean implementation to get the full dataset
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

def check_existing_cache():
    """Check if we already have NuminaMath cached."""
    cache_paths = [
        "data/numinamath_full.json",
        "data/numinamath_raw_full.json", 
        "data/datasets/numinamath_full.json"
    ]
    
    for path in cache_paths:
        if os.path.exists(path):
            print(f"📂 Found existing cache: {path}")
            with open(path, 'r') as f:
                data = json.load(f)
            print(f"✅ Cache contains {len(data)} examples")
            return path, data
    
    return None, None

def download_numinamath():
    """Download NuminaMath dataset from Hugging Face."""
    try:
        from datasets import load_dataset
        print("📥 Downloading NuminaMath from Hugging Face...")
        
        # Download the dataset
        dataset = load_dataset("AI-MO/NuminaMath-CoT", split="train")
        print(f"✅ Downloaded {len(dataset)} examples")
        
        # Convert to list of dictionaries
        data = []
        for i, example in enumerate(dataset):
            data.append({
                'id': f"numinamath_{i}",
                'problem': example.get('problem', ''),
                'solution': example.get('solution', ''),
                'source': example.get('source', ''),
                'subject': example.get('subject', ''),
                'difficulty': example.get('difficulty', 1)
            })
        
        return data
        
    except ImportError:
        print("❌ datasets library not found. Installing...")
        os.system("pip install datasets")
        return download_numinamath()
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

def save_dataset(data: List[Dict], filename: str):
    """Save dataset to file."""
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"💾 Saved {len(data)} examples to {output_path}")

def main():
    """Main function to download and cache NuminaMath dataset."""
    print("🚀 Getting NuminaMath dataset from Hugging Face")
    
    # Check if we already have it cached
    cache_path, cached_data = check_existing_cache()
    if cached_data:
        print(f"🎯 Using cached dataset with {len(cached_data)} examples")
        return cache_path
    
    # Download from Hugging Face
    data = download_numinamath()
    if not data:
        print("❌ Failed to download dataset")
        return None
    
    # Save to cache
    cache_file = "data/numinamath_full.json"
    save_dataset(data, cache_file)
    
    print(f"✅ NuminaMath dataset ready: {len(data)} examples")
    return cache_file

if __name__ == "__main__":
    result = main()
    if result:
        print(f"📁 Dataset available at: {result}")
    else:
        print("❌ Failed to get dataset")