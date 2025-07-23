"""
LateBench Dataset Manager
Handles loading, switching, and managing multiple datasets in unified format
"""

import json
import os
import shutil
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from data_processing.unified_schema import (
    LateBenchExample, create_timestamp
)


class LateBenchDatasetManager:
    """Manage multiple datasets and provide unified interface"""
    
    def __init__(self):
        self.datasets = {}  # dataset_name -> List[LateBenchExample]
        self.current_dataset = None
        self.current_examples = []
        self.manual_annotations = {}  # Combined annotations across datasets
        self.final_dataset = []  # Examples saved to final LateBench
        
        # File paths
        self.datasets_dir = "./data/datasets"
        self.annotations_file = "./data/annotations/manual_injection_data.json"
        self.final_file = "./data/final/latebench_v1.json"
        
        self._load_manual_annotations()
        self._load_final_dataset()
    
    def list_available_datasets(self) -> List[str]:
        """List all available dataset files"""
        if not os.path.exists(self.datasets_dir):
            return []
        
        datasets = []
        for file in os.listdir(self.datasets_dir):
            if file.startswith("latebench_") and file.endswith("_raw.json"):
                dataset_name = file.replace("latebench_", "").replace("_raw.json", "")
                datasets.append(dataset_name)
        
        return datasets
    
    def load_dataset(self, dataset_name: str) -> bool:
        """Load a specific dataset"""
        dataset_file = os.path.join(self.datasets_dir, f"latebench_{dataset_name}_raw.json")
        
        if not os.path.exists(dataset_file):
            print(f"Dataset file not found: {dataset_file}")
            return False
        
        try:
            print(f"Loading {dataset_name} dataset...")
            with open(dataset_file, 'r') as f:
                data = json.load(f)
            
            # Convert dictionaries back to LateBenchExample objects
            examples = []
            for item in data:
                try:
                    example = LateBenchExample.from_dict(item)
                    examples.append(example)
                except Exception as e:
                    print(f"Error loading example {item.get('id', 'unknown')}: {e}")
                    continue
            
            self.datasets[dataset_name] = examples
            self.current_dataset = dataset_name
            self.current_examples = examples
            
            print(f"✅ Loaded {len(examples)} examples from {dataset_name}")
            return True
            
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return False
    
    def switch_dataset(self, dataset_name: str) -> bool:
        """Switch to a different dataset"""
        if dataset_name in self.datasets:
            self.current_dataset = dataset_name
            self.current_examples = self.datasets[dataset_name]
            print(f"Switched to {dataset_name} dataset ({len(self.current_examples)} examples)")
            return True
        else:
            return self.load_dataset(dataset_name)
    
    def get_current_examples(self) -> List[LateBenchExample]:
        """Get examples from currently active dataset"""
        return self.current_examples
    
    def get_example_by_id(self, example_id: str) -> Optional[LateBenchExample]:
        """Get specific example by ID"""
        for example in self.current_examples:
            if example.id == example_id:
                return example
        return None
    
    def update_example(self, example_id: str, updated_example: LateBenchExample):
        """Update an example in the current dataset"""
        for i, example in enumerate(self.current_examples):
            if example.id == example_id:
                self.current_examples[i] = updated_example
                # Also update in the dataset cache
                if self.current_dataset in self.datasets:
                    self.datasets[self.current_dataset][i] = updated_example
                break
    
    def save_current_dataset(self):
        """Save current dataset back to file"""
        if not self.current_dataset or not self.current_examples:
            return False
        
        dataset_file = os.path.join(self.datasets_dir, f"latebench_{self.current_dataset}_raw.json")
        
        try:
            data = [example.to_dict() for example in self.current_examples]
            with open(dataset_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved {self.current_dataset} dataset")
            return True
        except Exception as e:
            print(f"Error saving dataset: {e}")
            return False
    
    def create_working_dataset(self, source_datasets: List[str], output_name: str = "working") -> bool:
        """Combine multiple datasets into a working dataset"""
        combined_examples = []
        
        for dataset_name in source_datasets:
            if dataset_name not in self.datasets:
                if not self.load_dataset(dataset_name):
                    continue
            
            examples = self.datasets[dataset_name]
            combined_examples.extend(examples)
        
        if not combined_examples:
            print("No examples found to combine")
            return False
        
        # Save combined dataset
        output_file = os.path.join(self.datasets_dir, f"latebench_{output_name}.json")
        data = [example.to_dict() for example in combined_examples]
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Load as current dataset
        self.datasets[output_name] = combined_examples
        self.current_dataset = output_name
        self.current_examples = combined_examples
        
        print(f"✅ Created working dataset '{output_name}' with {len(combined_examples)} examples")
        return True
    
    def _load_manual_annotations(self):
        """Load manual injection annotations"""
        if os.path.exists(self.annotations_file):
            try:
                with open(self.annotations_file, 'r') as f:
                    self.manual_annotations = json.load(f)
                print(f"Loaded manual annotations for {len(self.manual_annotations)} examples")
            except Exception as e:
                print(f"Error loading manual annotations: {e}")
                self.manual_annotations = {}
    
    def save_manual_annotations(self):
        """Save manual injection annotations"""
        try:
            os.makedirs(os.path.dirname(self.annotations_file), exist_ok=True)
            with open(self.annotations_file, 'w') as f:
                json.dump(self.manual_annotations, f, indent=2)
        except Exception as e:
            print(f"Error saving manual annotations: {e}")
    
    def get_manual_data(self, example_id: str) -> Dict[str, Any]:
        """Get manual annotation data for an example"""
        return self.manual_annotations.get(example_id, {
            "custom_suggestions": [],
            "injection_attempts": [],
            "final_decision": None,
            "decision_timestamp": None
        })
    
    def update_manual_data(self, example_id: str, manual_data: Dict[str, Any]):
        """Update manual annotation data for an example"""
        self.manual_annotations[example_id] = manual_data
        self.save_manual_annotations()
    
    def _load_final_dataset(self):
        """Load the final LateBench dataset"""
        if os.path.exists(self.final_file):
            try:
                with open(self.final_file, 'r') as f:
                    data = json.load(f)
                
                self.final_dataset = []
                for item in data:
                    try:
                        example = LateBenchExample.from_dict(item)
                        self.final_dataset.append(example)
                    except Exception as e:
                        print(f"Error loading final example: {e}")
                        continue
                
                print(f"Loaded {len(self.final_dataset)} examples from final LateBench dataset")
            except Exception as e:
                print(f"Error loading final dataset: {e}")
                self.final_dataset = []
    
    def save_to_final_dataset(self, example_id: str) -> bool:
        """Save an example to the final LateBench dataset"""
        example = self.get_example_by_id(example_id)
        if not example:
            return False
        
        # Check if already in final dataset
        for final_example in self.final_dataset:
            if final_example.id == example_id:
                # Update existing
                final_example = example
                print(f"Updated example {example_id} in final dataset")
                break
        else:
            # Add new
            self.final_dataset.append(example)
            print(f"Added example {example_id} to final dataset")
        
        # Update processing status
        example.processing.status = "finalized"
        example.processing.last_modified = create_timestamp()
        
        # Save final dataset
        try:
            os.makedirs(os.path.dirname(self.final_file), exist_ok=True)
            data = [example.to_dict() for example in self.final_dataset]
            with open(self.final_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Saved final LateBench dataset with {len(self.final_dataset)} examples")
            return True
            
        except Exception as e:
            print(f"Error saving to final dataset: {e}")
            return False
    
    def get_dataset_stats(self, dataset_name: str = None) -> Dict[str, Any]:
        """Get statistics for a dataset"""
        if dataset_name:
            examples = self.datasets.get(dataset_name, [])
        else:
            examples = self.current_examples
        
        if not examples:
            return {}
        
        stats = {
            "total_examples": len(examples),
            "datasets": {},
            "difficulties": {},
            "subjects": {},
            "step_counts": [],
            "importance_distribution": {"high": 0, "medium": 0, "low": 0},
            "decision_status": {"yes": 0, "maybe": 0, "no": 0, "undecided": 0}
        }
        
        for example in examples:
            # Source dataset
            source = example.source.dataset
            stats["datasets"][source] = stats["datasets"].get(source, 0) + 1
            
            # Difficulty
            diff = str(example.source.difficulty)
            stats["difficulties"][diff] = stats["difficulties"].get(diff, 0) + 1
            
            # Subject
            subj = example.source.subject
            stats["subjects"][subj] = stats["subjects"].get(subj, 0) + 1
            
            # Step count
            stats["step_counts"].append(example.solution.total_steps)
            
            # Step importance
            for step in example.solution.steps:
                stats["importance_distribution"][step.importance] += 1
            
            # Decision status
            decision = example.error_injection.final_decision
            if decision:
                stats["decision_status"][decision] += 1
            else:
                stats["decision_status"]["undecided"] += 1
        
        # Calculate averages
        if stats["step_counts"]:
            stats["avg_steps"] = sum(stats["step_counts"]) / len(stats["step_counts"])
            stats["step_range"] = [min(stats["step_counts"]), max(stats["step_counts"])]
        
        return stats
    
    def filter_examples(self, 
                       difficulty: Optional[Union[int, float, str]] = None,
                       subject: Optional[str] = None,
                       dataset: Optional[str] = None,
                       min_steps: Optional[int] = None,
                       max_steps: Optional[int] = None,
                       decision_filter: Optional[str] = None) -> List[LateBenchExample]:
        """Filter current examples based on criteria"""
        filtered = self.current_examples
        
        if difficulty:
            filtered = [ex for ex in filtered if str(ex.source.difficulty) == str(difficulty)]
        
        if subject:
            filtered = [ex for ex in filtered if ex.source.subject == subject]
        
        if dataset:
            filtered = [ex for ex in filtered if ex.source.dataset == dataset]
        
        if min_steps:
            filtered = [ex for ex in filtered if ex.solution.total_steps >= min_steps]
        
        if max_steps:
            filtered = [ex for ex in filtered if ex.solution.total_steps <= max_steps]
        
        if decision_filter:
            if decision_filter == "hide_no":
                filtered = [ex for ex in filtered if ex.error_injection.final_decision != "no"]
            elif decision_filter in ["yes", "maybe", "no"]:
                filtered = [ex for ex in filtered if ex.error_injection.final_decision == decision_filter]
            elif decision_filter == "undecided":
                filtered = [ex for ex in filtered if ex.error_injection.final_decision is None]
        
        return filtered