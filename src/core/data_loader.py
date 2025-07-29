"""
Unified data loading system for LateBench.
Core functionality: load and manage mathematical reasoning datasets using LateBenchExample objects.
"""

import os
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
from unified_schema import LateBenchExample


class LateBenchDataLoader:
    """Load and manage LateBench datasets returning LateBenchExample objects."""
    
    def __init__(self, datasets_dir: str = "./data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache loaded datasets as LateBenchExample objects
        self._dataset_cache: Dict[str, List[LateBenchExample]] = {}

    def list_available_datasets(self) -> Dict[str, List[str]]:
        """List all available dataset files organized by type."""
        datasets = {}
        
        if not self.datasets_dir.exists():
            return datasets
        
        for file in self.datasets_dir.glob("latebench_*.json"):
            name = file.stem.replace("latebench_", "")
            
            if "_complete" in name:
                base_name = name.replace("_complete", "")
                datasets.setdefault(base_name, []).append("complete")
            elif "_errors" in name:
                base_name = name.replace("_errors", "")
                datasets.setdefault(base_name, []).append("errors")
            elif "_raw" in name:
                base_name = name.replace("_raw", "")
                datasets.setdefault(base_name, []).append("all")
        
        return datasets

    def load_dataset(self, dataset_name: str, problem_type: str = "all") -> List[LateBenchExample]:
        """Load a dataset returning LateBenchExample objects."""
        cache_key = f"{dataset_name}_{problem_type}"
        
        if cache_key in self._dataset_cache:
            return self._dataset_cache[cache_key]
        
        # Determine file path
        if problem_type == "complete":
            filename = f"latebench_{dataset_name}_complete.json"
        elif problem_type == "errors":
            filename = f"latebench_{dataset_name}_errors.json"
        else:  # "all"
            filename = f"latebench_{dataset_name}_raw.json"
        
        filepath = self.datasets_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert to LateBenchExample objects
            examples = []
            for item in data:
                if isinstance(item, dict):
                    try:
                        example = LateBenchExample.from_dict(item)
                        examples.append(example)
                    except Exception as e:
                        print(f"Warning: Failed to parse example, skipping: {e}")
                else:
                    print(f"Warning: Invalid example format, skipping")
            
            self._dataset_cache[cache_key] = examples
            print(f"Loaded {len(examples)} examples from {dataset_name} ({problem_type})")
            
            return examples
            
        except Exception as e:
            raise RuntimeError(f"Error loading dataset {dataset_name}: {e}")

    def get_example_by_id(self, dataset_name: str, example_id: str, problem_type: str = "all") -> Optional[LateBenchExample]:
        """Get a specific example by ID."""
        examples = self.load_dataset(dataset_name, problem_type)
        
        for example in examples:
            if example.id == example_id:
                return example
        
        return None

    def save_dataset(self, examples: List[LateBenchExample], dataset_name: str, problem_type: str = "all"):
        """Save LateBenchExample objects to a dataset file."""
        if problem_type == "complete":
            filename = f"latebench_{dataset_name}_complete.json"
        elif problem_type == "errors":
            filename = f"latebench_{dataset_name}_errors.json"
        else:  # "all"
            filename = f"latebench_{dataset_name}_raw.json"
        
        filepath = self.datasets_dir / filename
        
        try:
            # Convert LateBenchExample objects to dictionaries for JSON serialization
            data = [example.to_dict() for example in examples]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Update cache
            cache_key = f"{dataset_name}_{problem_type}"
            self._dataset_cache[cache_key] = examples
            
            print(f"Saved {len(examples)} examples to {filename}")
            
        except Exception as e:
            raise RuntimeError(f"Error saving dataset {dataset_name}: {e}")

    def get_dataset_stats(self, dataset_name: str, problem_type: str = "all") -> Dict[str, Any]:
        """Get statistics for a dataset using LateBenchExample objects."""
        examples = self.load_dataset(dataset_name, problem_type)
        
        if not examples:
            return {}
        
        stats = {
            "total_examples": len(examples),
            "sources": {},
            "subjects": {},
            "difficulties": {},
            "step_counts": [],
            "error_types": {"has_errors": 0, "no_errors": 0}
        }
        
        for example in examples:
            # Source information using dataclass attributes
            dataset_source = example.source.dataset
            stats["sources"][dataset_source] = stats["sources"].get(dataset_source, 0) + 1
            
            # Subject
            subject = example.source.subject
            stats["subjects"][subject] = stats["subjects"].get(subject, 0) + 1
            
            # Difficulty
            difficulty = str(example.source.difficulty)
            stats["difficulties"][difficulty] = stats["difficulties"].get(difficulty, 0) + 1
            
            # Step count using dataclass attributes
            step_count = len(example.solution.steps)
            stats["step_counts"].append(step_count)
            
            # Error status using dataclass attributes
            has_errors = (
                example.source.metadata.get('has_errors', False) or
                example.error_injection.has_errors
            )
            
            if has_errors:
                stats["error_types"]["has_errors"] += 1
            else:
                stats["error_types"]["no_errors"] += 1
        
        # Calculate averages
        if stats["step_counts"]:
            stats["avg_steps"] = sum(stats["step_counts"]) / len(stats["step_counts"])
            stats["min_steps"] = min(stats["step_counts"])
            stats["max_steps"] = max(stats["step_counts"])
        
        return stats

    def filter_examples(self, examples: List[LateBenchExample], **filters) -> List[LateBenchExample]:
        """Filter LateBenchExample objects based on criteria."""
        filtered = examples
        
        if 'subject' in filters and filters['subject']:
            filtered = [ex for ex in filtered if ex.source.subject == filters['subject']]
        
        if 'min_steps' in filters and filters['min_steps']:
            filtered = [ex for ex in filtered if len(ex.solution.steps) >= filters['min_steps']]
        
        if 'max_steps' in filters and filters['max_steps']:
            filtered = [ex for ex in filtered if len(ex.solution.steps) <= filters['max_steps']]
        
        if 'has_errors' in filters:
            filtered = [ex for ex in filtered if self._example_has_errors(ex) == filters['has_errors']]
        
        return filtered

    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear dataset cache."""
        if dataset_name:
            keys_to_remove = [k for k in self._dataset_cache.keys() if k.startswith(dataset_name)]
            for key in keys_to_remove:
                del self._dataset_cache[key]
        else:
            self._dataset_cache.clear()


    def _example_has_errors(self, example: LateBenchExample) -> bool:
        """Check if LateBenchExample has errors."""
        return (
            example.source.metadata.get('has_errors', False) or
            example.error_injection.has_errors
        )


# Convenience functions
def load_dataset(dataset_name: str, problem_type: str = "all") -> List[LateBenchExample]:
    """Quick dataset loading returning LateBenchExample objects."""
    loader = LateBenchDataLoader()
    return loader.load_dataset(dataset_name, problem_type)


def get_dataset_stats(dataset_name: str, problem_type: str = "all") -> Dict[str, Any]:
    """Quick dataset statistics."""
    loader = LateBenchDataLoader()
    return loader.get_dataset_stats(dataset_name, problem_type)