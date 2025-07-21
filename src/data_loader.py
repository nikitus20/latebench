"""
Data loading and analysis utilities for the NuminaMath dataset.
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import json


class NuminaMathDataLoader:
    """Load and analyze the NuminaMath dataset for adversarial error injection."""
    
    def __init__(self, cache_dir: str = "./data"):
        self.cache_dir = cache_dir
        self.dataset = None
        self.local_path = os.path.join(cache_dir, "numinamath_local")
        
    def download_dataset(self) -> DatasetDict:
        """Download the NuminaMath dataset from HuggingFace."""
        print("Loading NuminaMath-CoT dataset...")
        
        try:
            # Try to load from local cache first
            if os.path.exists(self.local_path):
                print(f"Loading from local cache: {self.local_path}")
                from datasets import DatasetDict
                dataset = DatasetDict.load_from_disk(self.local_path)
            else:
                print("Downloading from HuggingFace...")
                dataset = load_dataset("AI-MO/NuminaMath-CoT")
                
                # Ensure cache directory exists
                os.makedirs(self.cache_dir, exist_ok=True)
                
                # Save locally for offline work
                print(f"Saving to local cache: {self.local_path}")
                dataset.save_to_disk(self.local_path)
                
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
            
        self.dataset = dataset
        return dataset
    
    def analyze_dataset_structure(self, dataset: Optional[DatasetDict] = None) -> Dict[str, Any]:
        """Analyze the structure and properties of the dataset."""
        if dataset is None:
            dataset = self.dataset
            
        if dataset is None:
            raise ValueError("No dataset loaded. Call download_dataset() first.")
        
        # Handle different dataset structures
        if isinstance(dataset, dict):
            # If it's a DatasetDict, use the train split
            data = dataset.get('train', list(dataset.values())[0])
        else:
            data = dataset
        
        analysis = {
            'total_examples': len(data),
            'features': list(data.features.keys()),
            'sample_example': data[0] if len(data) > 0 else None,
        }
        
        # Analyze solution lengths
        print("Analyzing solution lengths...")
        solution_lengths = []
        problem_types = []
        difficulty_levels = []
        
        for i, example in enumerate(tqdm(data, desc="Analyzing examples")):
            # Parse solution steps
            if 'solution' in example:
                steps = self.parse_solution_steps(example['solution'])
                solution_lengths.append(len(steps))
            
            # Collect problem types if available
            if 'type' in example and example['type']:
                problem_types.append(example['type'])
            
            # Collect difficulty levels if available
            if 'level' in example and example['level']:
                difficulty_levels.append(example['level'])
        
        analysis.update({
            'solution_statistics': {
                'mean_length': np.mean(solution_lengths),
                'median_length': np.median(solution_lengths),
                'std_length': np.std(solution_lengths),
                'min_length': min(solution_lengths),
                'max_length': max(solution_lengths),
                'length_distribution': np.histogram(solution_lengths, bins=20)[0].tolist()
            },
            'problem_type_counts': pd.Series(problem_types).value_counts().to_dict() if problem_types else {},
            'difficulty_level_counts': pd.Series(difficulty_levels).value_counts().to_dict() if difficulty_levels else {}
        })
        
        return analysis
    
    def parse_solution_steps(self, solution: str) -> List[str]:
        """Parse solution into individual steps."""
        if not solution or not isinstance(solution, str):
            return []
        
        # Handle various step formats
        steps = []
        lines = solution.strip().split('\n')
        current_step = []
        
        step_indicators = ['step', 'Step', 'STEP', '1.', '2.', '3.', '4.', '5.']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new step
            is_new_step = False
            if any(line.startswith(indicator) for indicator in step_indicators):
                is_new_step = True
            elif line[0].isdigit() and ('.' in line[:5] or ')' in line[:5]):
                is_new_step = True
            
            if is_new_step and current_step:
                steps.append(' '.join(current_step))
                current_step = [line]
            elif is_new_step:
                current_step = [line]
            else:
                current_step.append(line)
        
        # Don't forget the last step
        if current_step:
            steps.append(' '.join(current_step))
        
        # If no clear step structure found, split by sentences/paragraphs
        if len(steps) <= 1:
            # Split by double newlines (paragraphs)
            paragraphs = solution.split('\n\n')
            if len(paragraphs) > 1:
                steps = [p.strip() for p in paragraphs if p.strip()]
            else:
                # Split by sentences
                import re
                sentences = re.split(r'[.!?]+', solution)
                steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return [step for step in steps if step.strip()]
    
    def filter_long_solutions(self, dataset: Optional[Dataset] = None, 
                            min_steps: int = 8) -> List[Dict[str, Any]]:
        """Filter dataset to only include problems with sufficiently long solutions."""
        if dataset is None:
            if self.dataset is None:
                raise ValueError("No dataset loaded. Call download_dataset() first.")
            
            if isinstance(self.dataset, dict):
                dataset = self.dataset.get('train', list(self.dataset.values())[0])
            else:
                dataset = self.dataset
        
        filtered_examples = []
        
        print(f"Filtering examples with at least {min_steps} steps...")
        
        for example in tqdm(dataset, desc="Filtering examples"):
            if 'solution' not in example:
                continue
                
            steps = self.parse_solution_steps(example['solution'])
            
            if len(steps) >= min_steps:
                filtered_example = dict(example)
                filtered_example.update({
                    'parsed_steps': steps,
                    'num_steps': len(steps)
                })
                filtered_examples.append(filtered_example)
        
        print(f"Found {len(filtered_examples)} examples with >= {min_steps} steps")
        return filtered_examples
    
    def save_filtered_dataset(self, filtered_examples: List[Dict[str, Any]], 
                            filename: str = "filtered_long_solutions.json"):
        """Save filtered examples to disk."""
        filepath = os.path.join(self.cache_dir, filename)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(filtered_examples, f, indent=2)
        
        print(f"Saved {len(filtered_examples)} filtered examples to {filepath}")
    
    def load_filtered_dataset(self, filename: str = "filtered_long_solutions.json") -> List[Dict[str, Any]]:
        """Load filtered examples from disk."""
        filepath = os.path.join(self.cache_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"File {filepath} not found. Run filtering first.")
            return []
        
        with open(filepath, 'r') as f:
            examples = json.load(f)
        
        print(f"Loaded {len(examples)} filtered examples from {filepath}")
        return examples
    
    def get_sample_examples(self, n: int = 5, min_steps: int = 8) -> List[Dict[str, Any]]:
        """Get a sample of examples for testing/development."""
        filtered = self.filter_long_solutions(min_steps=min_steps)
        
        if len(filtered) < n:
            print(f"Only {len(filtered)} examples available with >= {min_steps} steps")
            return filtered
        
        # Sample evenly across different solution lengths
        filtered.sort(key=lambda x: x['num_steps'])
        indices = np.linspace(0, len(filtered) - 1, n).astype(int)
        
        return [filtered[i] for i in indices]