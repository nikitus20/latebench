"""
Simplified storage utilities for LateBench.
Core functionality: save and load results with basic organization and backup.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class LateBenchStorage:
    """Simple storage system for LateBench results and data."""
    
    def __init__(self, base_dir: str = "./data"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results"
        self.backups_dir = self.base_dir / "backups"
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def save_results(self, results: List[Any], experiment_name: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save experiment results with timestamp and metadata."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_dir = self.results_dir / f"{experiment_name}_{timestamp}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        results_file = experiment_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self._serialize_results(results), f, indent=2)
        
        # Save metadata
        if metadata:
            metadata_file = experiment_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create summary
        summary = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'total_results': len(results),
            'successful_results': sum(1 for r in results if self._is_successful(r)),
            'results_file': str(results_file),
            'metadata_file': str(metadata_file) if metadata else None
        }
        
        summary_file = experiment_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Saved {len(results)} results to {experiment_dir}")
        return str(experiment_dir)

    def load_results(self, experiment_path: str) -> tuple[List[Any], Dict[str, Any]]:
        """Load results and metadata from experiment directory."""
        
        experiment_dir = Path(experiment_path)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
        
        # Load results
        results_file = experiment_dir / "results.json"
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Load metadata if exists
        metadata = {}
        metadata_file = experiment_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        return results, metadata

    def list_experiments(self, experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments or experiments matching a name pattern."""
        
        experiments = []
        
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            if experiment_name and not exp_dir.name.startswith(experiment_name):
                continue
            
            summary_file = exp_dir / "summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    summary['path'] = str(exp_dir)
                    experiments.append(summary)
                except:
                    # Skip corrupted summaries
                    continue
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return experiments

    def save_critic_results(self, critic_results: Dict[str, Any], 
                           dataset_name: str, model_version: str) -> str:
        """Save critic evaluation results."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"critic_{dataset_name}_{model_version}_{timestamp}.json"
        filepath = self.results_dir / "critic" / filename
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for storage
        storage_data = {
            'dataset_name': dataset_name,
            'model_version': model_version,
            'timestamp': timestamp,
            'total_results': len(critic_results),
            'results': {}
        }
        
        # Convert critic results to serializable format
        for example_id, result in critic_results.items():
            if hasattr(result, 'to_dict'):
                storage_data['results'][example_id] = result.to_dict()
            else:
                storage_data['results'][example_id] = result
        
        with open(filepath, 'w') as f:
            json.dump(storage_data, f, indent=2)
        
        print(f"✅ Saved critic results to {filepath}")
        return str(filepath)

    def load_critic_results(self, dataset_name: str, 
                           model_version: Optional[str] = None) -> Dict[str, Any]:
        """Load latest critic results for a dataset."""
        
        critic_dir = self.results_dir / "critic"
        if not critic_dir.exists():
            return {}
        
        # Find matching files
        pattern = f"critic_{dataset_name}_"
        if model_version:
            pattern += f"{model_version}_"
        
        matching_files = [f for f in critic_dir.glob("*.json") if f.name.startswith(pattern)]
        
        if not matching_files:
            return {}
        
        # Get most recent file
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
            return data.get('results', {})
        except:
            return {}

    def create_backup(self, experiment_path: str) -> str:
        """Create backup of experiment directory."""
        
        experiment_dir = Path(experiment_path)
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_path}")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{experiment_dir.name}_backup_{timestamp}"
        backup_path = self.backups_dir / backup_name
        
        shutil.copytree(experiment_dir, backup_path)
        
        print(f"✅ Created backup: {backup_path}")
        return str(backup_path)

    def cleanup_old_results(self, days: int = 30):
        """Remove results older than specified days."""
        
        cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)
        removed_count = 0
        
        for exp_dir in self.results_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.stat().st_mtime < cutoff_time:
                # Create backup before deletion
                try:
                    self.create_backup(str(exp_dir))
                    shutil.rmtree(exp_dir)
                    removed_count += 1
                except:
                    continue
        
        print(f"✅ Cleaned up {removed_count} old experiment directories")

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        
        stats = {
            'total_experiments': 0,
            'total_size_mb': 0.0,
            'experiment_types': {},
            'recent_experiments': []
        }
        
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            stats['total_experiments'] += 1
            
            # Calculate size
            size = sum(f.stat().st_size for f in exp_dir.rglob('*') if f.is_file())
            stats['total_size_mb'] += size / (1024 * 1024)
            
            # Extract experiment type
            exp_type = exp_dir.name.split('_')[0] if '_' in exp_dir.name else 'unknown'
            stats['experiment_types'][exp_type] = stats['experiment_types'].get(exp_type, 0) + 1
        
        # Get recent experiments
        experiments = self.list_experiments()
        stats['recent_experiments'] = experiments[:5]  # Last 5
        
        return stats

    def _serialize_results(self, results: List[Any]) -> List[Any]:
        """Convert results to JSON-serializable format."""
        
        serialized = []
        for result in results:
            if hasattr(result, '__dict__'):
                # Convert dataclass/object to dict
                serialized.append(result.__dict__)
            elif hasattr(result, 'to_dict'):
                # Use custom serialization method
                serialized.append(result.to_dict())
            else:
                # Already serializable
                serialized.append(result)
        
        return serialized

    def _is_successful(self, result: Any) -> bool:
        """Check if result indicates success."""
        if hasattr(result, 'success'):
            return result.success
        elif isinstance(result, dict):
            return result.get('success', True)
        else:
            return True


# Convenience functions
def save_results(results: List[Any], experiment_name: str, 
                metadata: Optional[Dict[str, Any]] = None) -> str:
    """Quick save results."""
    storage = LateBenchStorage()
    return storage.save_results(results, experiment_name, metadata)


def load_results(experiment_path: str) -> tuple[List[Any], Dict[str, Any]]:
    """Quick load results."""
    storage = LateBenchStorage()
    return storage.load_results(experiment_path)


def list_experiments(experiment_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Quick list experiments."""
    storage = LateBenchStorage()
    return storage.list_experiments(experiment_name)


def save_examples_to_file(examples: List[Any], filepath: str) -> None:
    """Save LateBench examples to JSON file."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
    
    try:
        from unified_schema import LateBenchExample
    except ImportError:
        # Handle import differently if needed
        pass
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert examples to dictionaries
    serialized_examples = []
    for example in examples:
        if hasattr(example, 'to_dict'):
            serialized_examples.append(example.to_dict())
        else:
            serialized_examples.append(example)
    
    with open(filepath, 'w') as f:
        json.dump(serialized_examples, f, indent=2)
    
    print(f"💾 Saved {len(examples)} examples to {filepath}")


def load_examples_from_file(filepath: str) -> List[Any]:
    """Load LateBench examples from JSON file."""
    import sys
    from pathlib import Path
    
    # Add src to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
    
    try:
        from unified_schema import LateBenchExample
    except ImportError:
        raise ImportError("Could not import LateBenchExample from unified_schema")
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert dictionaries back to LateBenchExample objects
    examples = []
    for item in data:
        if isinstance(item, dict):
            examples.append(LateBenchExample.from_dict(item))
        else:
            examples.append(item)
    
    print(f"📂 Loaded {len(examples)} examples from {filepath}")
    return examples