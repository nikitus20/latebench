"""
Utility functions for the dashboard application.
"""

import json
import os
from typing import List, Dict, Any, Optional
import sys
import os
# Add both parent directory and src directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from core.critic import MathCritic
from core.data_loader import LateBenchDataLoader

# Simple compatibility classes for old dashboard
class StepFormatter:
    @staticmethod
    def clean_latex_escaping(text):
        """Simple text cleaning for dashboard display."""
        if not text:
            return ""
        # Remove extra LaTeX escaping
        text = str(text).replace('\\\\', '\\')
        return text

class CriticResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def to_dict(self):
        return self.__dict__

# Use LateBenchDataLoader as replacement for LateBenchDatasetManager
LateBenchDatasetManager = LateBenchDataLoader


class DashboardData:
    """Manages data loading and processing for the dashboard."""
    
    def __init__(self, results_file: str = "./data/small_experiment_results.json"):
        self.results_file = results_file
        self.examples = []
        self.critic_results = {}
        self.manual_injection_data = {}  # Store manual injection data
        self.dataset_manager = LateBenchDatasetManager()  # LateBenchDataLoader
        self.current_dataset_name = None
        self.current_problem_type = "all"
        self.load_data()
        self.load_manual_injection_data()
    
    def load_data(self):
        """Load examples using the new dataset manager or fallback to legacy loading."""
        # Try to load using dataset manager first
        try:
            available_datasets = self.dataset_manager.list_available_datasets()
        except:
            available_datasets = {}
        
        if available_datasets:
            # Prioritize MATH Level 5 natural errors dataset if available
            if 'math_level5_natural_errors' in available_datasets:
                first_dataset = 'math_level5_natural_errors'
                first_type = available_datasets[first_dataset][0] if available_datasets[first_dataset] else "all"
            # Fallback to old Level 5 dataset if available
            elif 'prm800k_level5_late' in available_datasets and 'errors' in available_datasets['prm800k_level5_late']:
                first_dataset = 'prm800k_level5_late'
                first_type = 'errors'
            else:
                # Load the first available dataset by default
                first_dataset = list(available_datasets.keys())[0]
                first_type = available_datasets[first_dataset][0] if available_datasets[first_dataset] else "all"
            
            try:
                examples = self.dataset_manager.load_dataset(first_dataset, first_type)
                if examples:
                    self.current_dataset_name = first_dataset
                    self.current_problem_type = first_type
                    self._convert_latebench_to_dashboard_format(examples)
                    return
            except:
                pass  # Fall back to legacy loading
        
        # Fallback to legacy loading
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                raw_data = json.load(f)
            
            # Process and clean data
            self.examples = []
            
            # Detect data format
            if raw_data and isinstance(raw_data[0], dict):
                if 'id' in raw_data[0] and 'source' in raw_data[0] and 'problem' in raw_data[0]:
                    # New LateBench unified format
                    print(f"Loading LateBench unified format data...")
                    for i, item in enumerate(raw_data):
                        example = self._process_latebench_example(item, i)
                        self.examples.append(example)
                    print(f"Loaded {len(self.examples)} LateBench examples")
                else:
                    # Old experiment format
                    for i, item in enumerate(raw_data):
                        if item.get('success', False):
                            example = self._process_example(item, i)
                            self.examples.append(example)
                    print(f"Loaded {len(self.examples)} successful adversarial examples")
            else:
                print("Warning: Unrecognized data format")
        else:
            print(f"No results file found at {self.results_file}")
    
    def _process_latebench_example(self, raw_example: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a LateBench unified format example for dashboard display."""
        
        # Extract basic information
        example_id = raw_example.get('id', f'example_{index}')
        source_info = raw_example.get('source', {})
        problem_info = raw_example.get('problem', {})
        solution_info = raw_example.get('solution', {})
        error_injection = raw_example.get('error_injection', {})
        
        # Convert to dashboard format
        processed = {
            'id': example_id,
            'index': index,
            'source': source_info.get('dataset', 'unknown'),
            'difficulty': source_info.get('difficulty', 3.0),
            'subject': source_info.get('subject', 'mathematics'),
            'problem': problem_info.get('statement', ''),
            'title': f"{source_info.get('dataset', 'unknown').title()} Problem {index + 1}",
            'solution': {
                'steps': solution_info.get('steps', []),
                'final_answer': solution_info.get('final_answer', ''),
                'total_steps': solution_info.get('total_steps', 0)
            },
            'has_errors': error_injection.get('has_errors', False),
            'metadata': source_info.get('metadata', {}),
            # Convert LateBench steps to dashboard format
            'original_steps': self._convert_latebench_steps(solution_info.get('steps', [])),
            'modified_steps': self._convert_latebench_steps(solution_info.get('steps', [])),  # Same for now
            # Add compatibility fields for old template references
            'error_info': {
                'type': source_info.get('dataset', 'unknown'),
                'step': self._find_first_error_step(solution_info.get('steps', [])),
                'what_changed': 'See step-by-step analysis below',
                'why_incorrect': 'Human-verified error identification available'
            },
            'original_solution': {
                'num_steps': solution_info.get('total_steps', 0)
            },
            'modified_solution': {
                'steps': solution_info.get('steps', [])
            }
        }
        
        return processed
    
    def _find_first_error_step(self, steps: List[Dict[str, Any]]) -> int:
        """Find the step number of the first error in LateBench steps."""
        for step in steps:
            if step.get('is_error', False) or step.get('is_modified', False):
                return step.get('step_number', 0)
        return 0  # No error found
    
    def _convert_latebench_steps(self, latebench_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert LateBench step format to dashboard step format."""
        converted_steps = []
        for step in latebench_steps:
            converted_step = {
                'step_number': step.get('step_number', 0),
                'content': step.get('content', ''),
                'reasoning_type': step.get('reasoning_type', 'unknown'),
                'importance': step.get('importance', 'medium'),
                'is_error': step.get('is_error', False),
                'is_modified': step.get('is_modified', False)
            }
            converted_steps.append(converted_step)
        return converted_steps
    
    def _process_example(self, raw_example: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process and clean a single example for dashboard display."""
        
        # Extract original problem
        original = raw_example.get('original_problem', {})
        
        # Extract modified solution
        modified = raw_example.get('modified_solution', {})
        
        # Extract error analysis
        error_analysis = raw_example.get('error_analysis', {})
        error_explanation = raw_example.get('error_explanation', {})
        
        # Clean and format steps
        original_steps = self._clean_steps(original.get('parsed_steps', []))
        modified_steps = self._clean_modified_steps(modified.get('steps', []))
        
        # Create processed example
        example = {
            'id': index,
            'title': self._generate_title(original.get('problem', '')),
            'problem': StepFormatter.clean_latex_escaping(original.get('problem', '')),
            'source': original.get('source', 'unknown'),
            
            # Original solution
            'original_solution': {
                'steps': original_steps,
                'answer': original.get('answer', 'No answer provided'),
                'num_steps': len(original_steps)
            },
            
            # Modified solution  
            'modified_solution': {
                'steps': modified_steps,
                'answer': StepFormatter.clean_latex_escaping(modified.get('final_answer', '')),
                'num_steps': len(modified_steps)
            },
            
            # Error information
            'error_info': {
                'type': error_analysis.get('error_type', 'unknown'),
                'step': error_analysis.get('selected_error_step', 0),
                'total_steps': error_analysis.get('total_steps', 0),
                'target_range': error_analysis.get('target_step_range', ''),
                'what_changed': StepFormatter.clean_latex_escaping(
                    error_explanation.get('what_changed', '')
                ),
                'why_incorrect': StepFormatter.clean_latex_escaping(
                    error_explanation.get('why_incorrect', '')
                ),
                'detection_difficulty': error_explanation.get('detection_difficulty', '')
            },
            
            # Metadata
            'metadata': raw_example.get('metadata', {}),
            
            # Placeholder for critic result
            'critic_result': None
        }
        
        return example
    
    def _clean_steps(self, steps: List[str]) -> List[Dict[str, Any]]:
        """Clean original solution steps."""
        cleaned_steps = []
        for i, step in enumerate(steps, 1):
            cleaned_steps.append({
                'number': i,
                'content': StepFormatter.clean_latex_escaping(str(step)),
                'is_modified': False,
                'is_error': False
            })
        return cleaned_steps
    
    def _clean_modified_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean modified solution steps with error indicators."""
        cleaned_steps = []
        for step in steps:
            cleaned_steps.append({
                'number': step.get('step_num', 0),
                'content': StepFormatter.clean_latex_escaping(step.get('content', '')),
                'is_modified': step.get('modified', False),
                'is_error': step.get('error', False)
            })
        return cleaned_steps
    
    def _generate_title(self, problem: str) -> str:
        """Generate a short title from the problem statement."""
        if not problem:
            return "Mathematical Problem"
        
        # Clean and truncate
        clean_problem = StepFormatter.clean_latex_escaping(problem)
        
        # Extract key terms
        if 'triangle' in clean_problem.lower():
            return "Triangle Problem"
        elif 'integral' in clean_problem.lower() or '∫' in clean_problem:
            return "Integration Problem"
        elif 'derivative' in clean_problem.lower():
            return "Differentiation Problem"
        elif 'equation' in clean_problem.lower():
            return "Equation Problem"
        elif 'probability' in clean_problem.lower():
            return "Probability Problem"
        else:
            # Use first few words
            words = clean_problem.split()[:4]
            return ' '.join(words) + "..."
    
    def get_example(self, example_id) -> Optional[Dict[str, Any]]:
        """Get example by ID, with manual injection data and critic results integrated."""
        for example in self.examples:
            if example['id'] == example_id:
                # Create a copy to avoid modifying the original
                example_copy = example.copy()
                
                # Check for manual injection data and integrate it
                manual_data = self.get_manual_injection_data(example_id)
                if manual_data.get('injection_attempts'):
                    # Get the most recent successful injection
                    for attempt in reversed(manual_data['injection_attempts']):
                        injection_result = attempt.get('injection_result', {})
                        if injection_result.get('success'):
                            # Integrate the injection result into the example
                            self._integrate_injection_result(example_copy, injection_result)
                            break
                
                # Check for critic results and integrate them
                if example_id in self.critic_results:
                    example_copy['critic_result'] = self.critic_results[example_id]
                
                return example_copy
        return None
    
    def get_examples_by_error_type(self, error_type: str) -> List[Dict[str, Any]]:
        """Get examples filtered by error type."""
        return [ex for ex in self.examples if ex['error_info']['type'] == error_type]
    
    def get_error_types(self) -> List[str]:
        """Get list of all error types in the data."""
        error_types = set()
        for example in self.examples:
            if 'error_info' in example:
                # Old format
                error_types.add(example['error_info']['type'])
            else:
                # New LateBench format - use source dataset as type
                error_types.add(example.get('source', 'unknown'))
        return sorted(list(error_types))
    
    def _integrate_injection_result(self, example: Dict[str, Any], injection_result: Dict[str, Any]):
        """Integrate manual injection result into example data."""
        modified_solution = injection_result.get('modified_solution', {})
        error_analysis = injection_result.get('error_analysis', {})
        error_explanation = injection_result.get('error_explanation', {})
        
        if modified_solution.get('steps'):
            # Store original step information for preservation
            original_steps = example.get('modified_steps', [])
            original_steps_dict = {step.get('step_number', step.get('number', i+1)): step 
                                 for i, step in enumerate(original_steps)}
            
            # Convert injection result format to dashboard format
            dashboard_steps = []
            for step in modified_solution['steps']:
                step_num = step.get('step_num', 0)
                is_modified = step.get('modified', False)
                is_error = step.get('error', False)
                
                # Get original step info if available
                original_step = original_steps_dict.get(step_num, {})
                original_reasoning_type = original_step.get('reasoning_type', 'unknown')
                original_importance = original_step.get('importance', 'medium')
                
                # Preserve original reasoning type for unmodified steps, use 'injected' only for modified ones
                reasoning_type = 'injected' if is_modified else original_reasoning_type
                
                # Determine importance: high for error steps, preserve original for unmodified, medium for other modified
                if is_error:
                    importance = 'high'
                elif not is_modified:
                    importance = original_importance
                else:
                    importance = 'medium'
                
                dashboard_steps.append({
                    'step_number': step_num,
                    'number': step_num,  # Alternative field name
                    'content': StepFormatter.clean_latex_escaping(step.get('content', '')),
                    'is_modified': is_modified,
                    'is_error': is_error,
                    'reasoning_type': reasoning_type,
                    'importance': importance,
                    # Add injection metadata without overriding original classification
                    'was_injected': True,  # Mark that this step went through injection process
                    'original_reasoning_type': original_reasoning_type,
                    'injection_modified': is_modified  # Track if this specific step was modified during injection
                })
            
            # Update the example's modified steps
            example['modified_steps'] = dashboard_steps
            example['modified_solution'] = {
                'steps': dashboard_steps,
                'answer': StepFormatter.clean_latex_escaping(modified_solution.get('final_answer', '')),
                'num_steps': len(dashboard_steps)
            }
            
            # Update error information with injection analysis
            if error_analysis:
                example['error_info'].update({
                    'step': error_analysis.get('selected_error_step', 0),
                    'type': error_analysis.get('error_type', 'injected'),
                    'total_steps': error_analysis.get('total_steps', 0),
                    'target_range': error_analysis.get('target_step_range', ''),
                    'what_changed': StepFormatter.clean_latex_escaping(
                        error_explanation.get('what_changed', '')
                    ),
                    'why_incorrect': StepFormatter.clean_latex_escaping(
                        error_explanation.get('why_incorrect', '')
                    )
                })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics using dataset manager when possible."""
        # Try to get enhanced stats from dataset manager
        if hasattr(self, 'dataset_manager') and self.dataset_manager.current_examples:
            stats = self.dataset_manager.get_dataset_stats()
            # Add dashboard-specific statistics
            stats.update({
                'unique_error_types': len(stats.get('datasets', {})),
                'current_dataset': self.get_current_dataset_info()
            })
            return stats
        
        # Fallback to legacy statistics
        if not self.examples:
            return {}
        
        error_type_counts = {}
        step_counts = []
        difficulty_counts = {}
        problem_type_counts = {'complete_solutions': 0, 'error_labeled': 0}
        
        for example in self.examples:
            # Use unified approach with compatibility fields
            error_type = example.get('error_info', {}).get('type', example.get('source', 'unknown'))
            step_count = example.get('original_solution', {}).get('num_steps', 
                                   example.get('solution', {}).get('total_steps', 0))
            
            # Determine difficulty
            if 'error_info' in example and 'detection_difficulty' in example['error_info']:
                # Old format with detection_difficulty
                diff_text = example['error_info']['detection_difficulty'].lower()
                if 'high' in diff_text:
                    difficulty = 'High'
                elif 'medium' in diff_text:
                    difficulty = 'Medium'
                elif 'low' in diff_text:
                    difficulty = 'Low'
                else:
                    difficulty = 'Unknown'
            else:
                # New LateBench format - use source difficulty
                raw_difficulty = example.get('difficulty', 3.0)
                if raw_difficulty >= 4.5:
                    difficulty = 'High'
                elif raw_difficulty >= 3.5:
                    difficulty = 'Medium'
                else:
                    difficulty = 'Low'
            
            # Problem type counting
            has_errors = example.get('metadata', {}).get('has_errors', False)
            if has_errors:
                problem_type_counts['error_labeled'] += 1
            else:
                problem_type_counts['complete_solutions'] += 1
            
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            step_counts.append(step_count)
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        return {
            'total_examples': len(self.examples),
            'error_types': error_type_counts,
            'avg_steps': sum(step_counts) / len(step_counts) if step_counts else 0,
            'step_range': [min(step_counts), max(step_counts)] if step_counts else [0, 0],
            'difficulty_distribution': difficulty_counts,
            'unique_error_types': len(error_type_counts),
            'error_status': problem_type_counts,
            'current_dataset': self.get_current_dataset_info()
        }
    
    def add_critic_result(self, example_id: int, critic_result: CriticResult):
        """Add critic evaluation result to an example."""
        example = self.get_example(example_id)
        if example:
            example['critic_result'] = critic_result.to_dict()
            
            # Store in separate dict for persistence
            self.critic_results[example_id] = critic_result.to_dict()
    
    def has_critic_results(self, example_id: int) -> bool:
        """Check if example has critic results."""
        example = self.get_example(example_id)
        return example and example.get('critic_result') is not None
    
    def save_critic_results(self, filepath: str = "./data/dashboard_critic_results.json"):
        """Save critic results to file."""
        if self.critic_results:
            with open(filepath, 'w') as f:
                json.dump(self.critic_results, f, indent=2)
            print(f"Saved critic results to {filepath}")
    
    def load_critic_results(self, filepath: str = "./data/dashboard_critic_results.json"):
        """Load critic results from file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.critic_results = json.load(f)
            
            # Apply to examples
            for example_id_str, result in self.critic_results.items():
                # Use the string ID directly (no conversion to int needed)
                example = self.get_example(example_id_str)
                if example:
                    example['critic_result'] = result
            
            print(f"Loaded critic results from {filepath}")
    
    def load_manual_injection_data(self, filepath: str = "./data/manual_injection_data.json"):
        """Load manual injection data from file."""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r') as f:
                    self.manual_injection_data = json.load(f)
                print(f"Loaded manual injection data from {filepath}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error loading manual injection data: {e}")
                self.manual_injection_data = {}
        else:
            print(f"No manual injection data file found at {filepath}")
            self.manual_injection_data = {}
    
    def save_manual_injection_data(self, filepath: str = "./data/manual_injection_data.json"):
        """Save manual injection data to file."""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.manual_injection_data, f, indent=2)
            print(f"Saved manual injection data to {filepath}")
        except (IOError, OSError) as e:
            print(f"Error saving manual injection data: {e}")
    
    def get_manual_injection_data(self, example_id: int) -> Dict[str, Any]:
        """Get manual injection data for a specific example."""
        return self.manual_injection_data.get(str(example_id), {
            "custom_suggestions": [],
            "injection_attempts": [],
            "final_decision": None,
            "decision_timestamp": None
        })
    
    def update_custom_suggestion(self, example_id: int, suggestion: str):
        """Add or update custom error suggestion for an example."""
        example_id_str = str(example_id)
        if example_id_str not in self.manual_injection_data:
            self.manual_injection_data[example_id_str] = {
                "custom_suggestions": [],
                "injection_attempts": [],
                "final_decision": None,
                "decision_timestamp": None
            }
        
        # Add suggestion if not already present
        if suggestion and suggestion not in self.manual_injection_data[example_id_str]["custom_suggestions"]:
            self.manual_injection_data[example_id_str]["custom_suggestions"].append(suggestion)
    
    def add_injection_attempt(self, example_id: int, attempt_data: Dict[str, Any]):
        """Add a manual injection attempt with remarks."""
        import time
        
        example_id_str = str(example_id)
        if example_id_str not in self.manual_injection_data:
            self.manual_injection_data[example_id_str] = {
                "custom_suggestions": [],
                "injection_attempts": [],
                "final_decision": None,
                "decision_timestamp": None
            }
        
        attempt = {
            "attempt_number": len(self.manual_injection_data[example_id_str]["injection_attempts"]) + 1,
            "user_remarks": attempt_data.get("user_remarks", ""),
            "injection_result": attempt_data.get("injection_result", {}),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        
        self.manual_injection_data[example_id_str]["injection_attempts"].append(attempt)
        
        # Limit to 2 attempts as specified
        if len(self.manual_injection_data[example_id_str]["injection_attempts"]) > 2:
            self.manual_injection_data[example_id_str]["injection_attempts"] = \
                self.manual_injection_data[example_id_str]["injection_attempts"][-2:]
    
    def set_final_decision(self, example_id: int, decision: str):
        """Set final decision (yes/maybe/no) for an example."""
        import time
        
        example_id_str = str(example_id)
        if example_id_str not in self.manual_injection_data:
            self.manual_injection_data[example_id_str] = {
                "custom_suggestions": [],
                "injection_attempts": [],
                "final_decision": None,
                "decision_timestamp": None
            }
        
        if decision in ["yes", "maybe", "no"]:
            self.manual_injection_data[example_id_str]["final_decision"] = decision
            self.manual_injection_data[example_id_str]["decision_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    def get_examples_by_decision(self, decision: Optional[str] = None, exclude_decision: Optional[str] = None) -> List[Dict[str, Any]]:
        """Filter examples by final decision status."""
        filtered_examples = []
        
        for example in self.examples:
            example_data = self.get_manual_injection_data(example['id'])
            current_decision = example_data.get('final_decision')
            
            # Apply filters
            if exclude_decision and current_decision == exclude_decision:
                continue
            
            if decision and current_decision != decision:
                continue
            
            filtered_examples.append(example)
        
        return filtered_examples
    
    def get_available_datasets(self) -> Dict[str, List[str]]:
        """Get available datasets from the dataset manager."""
        return self.dataset_manager.list_available_datasets()
    
    def switch_dataset(self, dataset_name: str, problem_type: str = "all") -> bool:
        """Switch to a different dataset and problem type."""
        try:
            examples = self.dataset_manager.load_dataset(dataset_name, problem_type)
            if examples:
                self.current_dataset_name = dataset_name
                self.current_problem_type = problem_type
                self._convert_latebench_to_dashboard_format(examples)
                return True
        except:
            pass
        return False
    
    def get_current_dataset_info(self) -> Dict[str, str]:
        """Get current dataset information."""
        return {
            "name": self.current_dataset_name or "None",
            "type": self.current_problem_type or "all"
        }
    
    def _convert_latebench_to_dashboard_format(self, latebench_examples):
        """Convert LateBench examples to dashboard format."""
        self.examples = []
        
        for i, lb_example in enumerate(latebench_examples):
            # Convert LateBench format to dashboard format
            example_dict = {
                "id": lb_example.id,
                "source": {
                    "dataset": lb_example.source.dataset,
                    "original_id": lb_example.source.original_id,
                    "difficulty": lb_example.source.difficulty,
                    "subject": lb_example.source.subject,
                    "competition": lb_example.source.competition,
                    "metadata": lb_example.source.metadata
                },
                "problem": {
                    "statement": lb_example.problem.statement
                },
                "solution": {
                    "steps": [{
                        "step_number": step.step_number,
                        "content": step.content,
                        "importance": step.importance,
                        "reasoning_type": step.reasoning_type,
                        "is_error": step.is_error,
                        "is_modified": step.is_modified
                    } for step in lb_example.solution.steps],
                    "final_answer": lb_example.solution.final_answer,
                    "total_steps": lb_example.solution.total_steps,
                    "solution_method": lb_example.solution.solution_method
                },
                "error_injection": {
                    "has_errors": lb_example.error_injection.has_errors
                },
                "processing": {
                    "added_to_latebench": lb_example.processing.added_to_latebench,
                    "last_modified": lb_example.processing.last_modified,
                    "status": lb_example.processing.status
                }
            }
            
            dashboard_example = self._process_latebench_example(example_dict, i)
            self.examples.append(dashboard_example)
        
        print(f"Converted {len(self.examples)} LateBench examples to dashboard format")


def analyze_critic_performance(example: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze critic performance on a single example."""
    
    if not example.get('critic_result'):
        return {'status': 'no_evaluation'}
    
    critic_result = example['critic_result']
    ground_truth_step = example['error_info']['step']
    
    # Check if critic found errors
    critic_found_errors = critic_result.get('has_errors', False)
    critic_error_steps = critic_result.get('error_steps', [])
    
    analysis = {
        'status': 'evaluated',
        'critic_found_errors': critic_found_errors,
        'ground_truth_error_step': ground_truth_step,
        'critic_error_steps': critic_error_steps,
        'correct_detection': False,
        'exact_step_match': False,
        'false_positives': [],
        'missed_error': False
    }
    
    if critic_found_errors:
        # Check if critic found the correct error step
        if ground_truth_step in critic_error_steps:
            analysis['correct_detection'] = True
            analysis['exact_step_match'] = True
        
        # Check for false positives (steps marked as errors that aren't)
        analysis['false_positives'] = [
            step for step in critic_error_steps 
            if step != ground_truth_step
        ]
    else:
        # Critic found no errors, but there is an injected error
        analysis['missed_error'] = True
    
    return analysis