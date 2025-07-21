"""
Utility functions for the dashboard application.
"""

import json
import os
from typing import List, Dict, Any, Optional
from critic import StepFormatter, CriticResult


class DashboardData:
    """Manages data loading and processing for the dashboard."""
    
    def __init__(self, results_file: str = "./data/small_experiment_results.json"):
        self.results_file = results_file
        self.examples = []
        self.critic_results = {}
        self.load_data()
    
    def load_data(self):
        """Load adversarial examples from file."""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                raw_data = json.load(f)
            
            # Process and clean data
            self.examples = []
            for i, item in enumerate(raw_data):
                if item.get('success', False):
                    example = self._process_example(item, i)
                    self.examples.append(example)
            
            print(f"Loaded {len(self.examples)} successful adversarial examples")
        else:
            print(f"No results file found at {self.results_file}")
    
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
    
    def get_example(self, example_id: int) -> Optional[Dict[str, Any]]:
        """Get example by ID."""
        for example in self.examples:
            if example['id'] == example_id:
                return example
        return None
    
    def get_examples_by_error_type(self, error_type: str) -> List[Dict[str, Any]]:
        """Get examples filtered by error type."""
        return [ex for ex in self.examples if ex['error_info']['type'] == error_type]
    
    def get_error_types(self) -> List[str]:
        """Get list of all error types in the data."""
        error_types = set()
        for example in self.examples:
            error_types.add(example['error_info']['type'])
        return sorted(list(error_types))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.examples:
            return {}
        
        error_type_counts = {}
        step_counts = []
        difficulty_counts = {}
        
        for example in self.examples:
            # Error types
            error_type = example['error_info']['type']
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
            
            # Step counts
            step_counts.append(example['original_solution']['num_steps'])
            
            # Difficulty (extract from detection difficulty)
            difficulty = 'Unknown'
            diff_text = example['error_info']['detection_difficulty'].lower()
            if 'high' in diff_text:
                difficulty = 'High'
            elif 'medium' in diff_text:
                difficulty = 'Medium'
            elif 'low' in diff_text:
                difficulty = 'Low'
            
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        return {
            'total_examples': len(self.examples),
            'error_types': error_type_counts,
            'avg_steps': sum(step_counts) / len(step_counts) if step_counts else 0,
            'step_range': [min(step_counts), max(step_counts)] if step_counts else [0, 0],
            'difficulty_distribution': difficulty_counts,
            'unique_error_types': len(error_type_counts)
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
                example_id = int(example_id_str)
                example = self.get_example(example_id)
                if example:
                    example['critic_result'] = result
            
            print(f"Loaded critic results from {filepath}")


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