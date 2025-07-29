"""
LateBench Dataset Filtering Utilities
Provides general-purpose filtering functions for LateBench datasets
"""

from typing import List, Dict, Any, Optional, Callable
from data_processing.unified_schema import LateBenchExample


class LateBenchFilter:
    """General-purpose filtering utilities for LateBench datasets"""
    
    @staticmethod
    def by_difficulty(examples: List[LateBenchExample], 
                     min_difficulty: Optional[float] = None,
                     max_difficulty: Optional[float] = None) -> List[LateBenchExample]:
        """Filter examples by difficulty range"""
        filtered = []
        for example in examples:
            difficulty = example.source.difficulty
            if isinstance(difficulty, (int, float)):
                if min_difficulty is not None and difficulty < min_difficulty:
                    continue
                if max_difficulty is not None and difficulty > max_difficulty:
                    continue
            filtered.append(example)
        return filtered
    
    @staticmethod
    def by_step_count(examples: List[LateBenchExample],
                     min_steps: Optional[int] = None,
                     max_steps: Optional[int] = None) -> List[LateBenchExample]:
        """Filter examples by total number of solution steps"""
        filtered = []
        for example in examples:
            step_count = example.solution.total_steps
            if min_steps is not None and step_count < min_steps:
                continue
            if max_steps is not None and step_count > max_steps:
                continue
            filtered.append(example)
        return filtered
    
    @staticmethod
    def by_error_status(examples: List[LateBenchExample],
                       has_errors: Optional[bool] = None) -> List[LateBenchExample]:
        """Filter examples by error status"""
        if has_errors is None:
            return examples
        
        return [ex for ex in examples if ex.error_injection.has_errors == has_errors]
    
    @staticmethod
    def by_dataset(examples: List[LateBenchExample],
                  datasets: List[str]) -> List[LateBenchExample]:
        """Filter examples by source dataset"""
        return [ex for ex in examples if ex.source.dataset in datasets]
    
    @staticmethod
    def by_subject(examples: List[LateBenchExample],
                  subjects: List[str]) -> List[LateBenchExample]:
        """Filter examples by subject"""
        return [ex for ex in examples if ex.source.subject in subjects]
    
    @staticmethod
    def by_competition(examples: List[LateBenchExample],
                      competitions: List[str]) -> List[LateBenchExample]:
        """Filter examples by competition"""
        filtered = []
        for example in examples:
            if example.source.competition and example.source.competition in competitions:
                filtered.append(example)
        return filtered
    
    @staticmethod
    def by_late_errors(examples: List[LateBenchExample],
                      min_error_step: int = 12) -> List[LateBenchExample]:
        """Filter for examples with late errors (error step >= min_error_step)
        
        For complete solutions: include all
        For error solutions: only include if first error is late enough
        """
        filtered = []
        for example in examples:
            if not example.error_injection.has_errors:
                # Complete solution - include
                filtered.append(example)
                continue
            
            # Find first error step
            first_error_step = None
            for step in example.solution.steps:
                if step.is_error:
                    first_error_step = step.step_number
                    break
            
            # Include if error is late enough
            if first_error_step and first_error_step >= min_error_step:
                filtered.append(example)
        
        return filtered
    
    @staticmethod
    def by_reasoning_types(examples: List[LateBenchExample],
                          reasoning_types: List[str]) -> List[LateBenchExample]:
        """Filter examples that contain specific reasoning types"""
        filtered = []
        for example in examples:
            # Check if any step has the required reasoning type
            if any(step.reasoning_type in reasoning_types for step in example.solution.steps):
                filtered.append(example)
        return filtered
    
    @staticmethod
    def by_importance_levels(examples: List[LateBenchExample],
                           importance_levels: List[str]) -> List[LateBenchExample]:
        """Filter examples that contain steps with specific importance levels"""
        filtered = []
        for example in examples:
            # Check if any step has the required importance level
            if any(step.importance in importance_levels for step in example.solution.steps):
                filtered.append(example)
        return filtered
    
    @staticmethod
    def by_critic_predictions(examples: List[LateBenchExample],
                             has_predictions: Optional[bool] = None,
                             critic_has_errors: Optional[bool] = None) -> List[LateBenchExample]:
        """Filter examples by critic prediction status"""
        filtered = []
        for example in examples:
            # Filter by whether predictions exist
            if has_predictions is not None:
                if (example.critic_predictions is not None) != has_predictions:
                    continue
            
            # Filter by critic's error prediction (only if predictions exist)
            if critic_has_errors is not None and example.critic_predictions:
                if example.critic_predictions.has_errors != critic_has_errors:
                    continue
            
            filtered.append(example)
        return filtered
    
    @staticmethod
    def by_critic_accuracy(examples: List[LateBenchExample],
                          correct_predictions_only: bool = True) -> List[LateBenchExample]:
        """Filter examples where critic predictions match ground truth"""
        filtered = []
        for example in examples:
            if not example.critic_predictions:
                continue  # Skip examples without predictions
            
            # Compare critic prediction with ground truth
            ground_truth_has_errors = example.error_injection.has_errors
            critic_prediction = example.critic_predictions.has_errors
            
            is_correct = (ground_truth_has_errors == critic_prediction)
            
            if correct_predictions_only and is_correct:
                filtered.append(example)
            elif not correct_predictions_only and not is_correct:
                filtered.append(example)
        
        return filtered
    
    @staticmethod
    def by_custom_criteria(examples: List[LateBenchExample],
                          filter_func: Callable[[LateBenchExample], bool]) -> List[LateBenchExample]:
        """Filter examples using a custom function"""
        return [ex for ex in examples if filter_func(ex)]


class LateBenchSorter:
    """Sorting utilities for LateBench datasets"""
    
    @staticmethod
    def by_difficulty(examples: List[LateBenchExample], 
                     reverse: bool = False) -> List[LateBenchExample]:
        """Sort examples by difficulty"""
        def get_difficulty(ex: LateBenchExample) -> float:
            difficulty = ex.source.difficulty
            if isinstance(difficulty, (int, float)):
                return float(difficulty)
            return 3.0  # Default for non-numeric difficulties
        
        return sorted(examples, key=get_difficulty, reverse=reverse)
    
    @staticmethod
    def by_step_count(examples: List[LateBenchExample],
                     reverse: bool = False) -> List[LateBenchExample]:
        """Sort examples by total step count"""
        return sorted(examples, key=lambda ex: ex.solution.total_steps, reverse=reverse)
    
    @staticmethod
    def by_error_step(examples: List[LateBenchExample],
                     reverse: bool = False) -> List[LateBenchExample]:
        """Sort examples by first error step position"""
        def get_first_error_step(ex: LateBenchExample) -> int:
            if not ex.error_injection.has_errors:
                return float('inf')  # Complete solutions go last
            
            for step in ex.solution.steps:
                if step.is_error:
                    return step.step_number
            return float('inf')  # No error found
        
        return sorted(examples, key=get_first_error_step, reverse=reverse)
    
    @staticmethod
    def by_dataset(examples: List[LateBenchExample],
                  reverse: bool = False) -> List[LateBenchExample]:
        """Sort examples by dataset name"""
        return sorted(examples, key=lambda ex: ex.source.dataset, reverse=reverse)


class LateBenchSplitter:
    """Dataset splitting utilities for LateBench datasets"""
    
    @staticmethod
    def by_error_status(examples: List[LateBenchExample]) -> Dict[str, List[LateBenchExample]]:
        """Split examples by error status"""
        complete_solutions = []
        error_solutions = []
        
        for example in examples:
            if example.error_injection.has_errors:
                error_solutions.append(example)
            else:
                complete_solutions.append(example)
        
        return {
            'complete': complete_solutions,
            'errors': error_solutions
        }
    
    @staticmethod
    def by_dataset(examples: List[LateBenchExample]) -> Dict[str, List[LateBenchExample]]:
        """Split examples by source dataset"""
        splits = {}
        for example in examples:
            dataset = example.source.dataset
            if dataset not in splits:
                splits[dataset] = []
            splits[dataset].append(example)
        return splits
    
    @staticmethod
    def by_difficulty_ranges(examples: List[LateBenchExample]) -> Dict[str, List[LateBenchExample]]:
        """Split examples by difficulty ranges (easy, medium, hard)"""
        easy = []  # difficulty < 2.5
        medium = []  # 2.5 <= difficulty < 4.0
        hard = []  # difficulty >= 4.0
        unknown = []  # non-numeric difficulty
        
        for example in examples:
            difficulty = example.source.difficulty
            if isinstance(difficulty, (int, float)):
                if difficulty < 2.5:
                    easy.append(example)
                elif difficulty < 4.0:
                    medium.append(example)
                else:
                    hard.append(example)
            else:
                unknown.append(example)
        
        return {
            'easy': easy,
            'medium': medium,
            'hard': hard,
            'unknown': unknown
        }
    
    @staticmethod
    def train_test_split(examples: List[LateBenchExample],
                        test_ratio: float = 0.2,
                        random_seed: Optional[int] = None) -> Dict[str, List[LateBenchExample]]:
        """Split examples into train/test sets"""
        import random
        
        if random_seed is not None:
            random.seed(random_seed)
        
        shuffled = examples.copy()
        random.shuffle(shuffled)
        
        test_size = int(len(shuffled) * test_ratio)
        
        return {
            'train': shuffled[test_size:],
            'test': shuffled[:test_size]
        }


def get_filtering_statistics(examples: List[LateBenchExample]) -> Dict[str, Any]:
    """Get comprehensive statistics about a LateBench dataset"""
    if not examples:
        return {}
    
    stats = {
        'total_examples': len(examples),
        'datasets': {},
        'subjects': {},
        'competitions': {},
        'difficulty_distribution': {'easy': 0, 'medium': 0, 'hard': 0, 'unknown': 0},
        'error_status': {'complete': 0, 'errors': 0},
        'step_counts': {'min': float('inf'), 'max': 0, 'avg': 0},
        'reasoning_types': {},
        'importance_levels': {},
        'critic_predictions': {
            'total_with_predictions': 0,
            'critic_says_errors': 0,
            'critic_says_clean': 0,
            'accuracy': {'correct': 0, 'incorrect': 0, 'total_comparable': 0}
        }
    }
    
    total_steps = 0
    
    for example in examples:
        # Dataset distribution
        dataset = example.source.dataset
        stats['datasets'][dataset] = stats['datasets'].get(dataset, 0) + 1
        
        # Subject distribution
        subject = example.source.subject
        stats['subjects'][subject] = stats['subjects'].get(subject, 0) + 1
        
        # Competition distribution
        if example.source.competition:
            comp = example.source.competition
            stats['competitions'][comp] = stats['competitions'].get(comp, 0) + 1
        
        # Difficulty distribution
        difficulty = example.source.difficulty
        if isinstance(difficulty, (int, float)):
            if difficulty < 2.5:
                stats['difficulty_distribution']['easy'] += 1
            elif difficulty < 4.0:
                stats['difficulty_distribution']['medium'] += 1
            else:
                stats['difficulty_distribution']['hard'] += 1
        else:
            stats['difficulty_distribution']['unknown'] += 1
        
        # Error status
        if example.error_injection.has_errors:
            stats['error_status']['errors'] += 1
        else:
            stats['error_status']['complete'] += 1
        
        # Step counts
        step_count = example.solution.total_steps
        stats['step_counts']['min'] = min(stats['step_counts']['min'], step_count)
        stats['step_counts']['max'] = max(stats['step_counts']['max'], step_count)
        total_steps += step_count
        
        # Reasoning types and importance levels
        for step in example.solution.steps:
            # Reasoning types
            rt = step.reasoning_type
            stats['reasoning_types'][rt] = stats['reasoning_types'].get(rt, 0) + 1
            
            # Importance levels
            imp = step.importance
            stats['importance_levels'][imp] = stats['importance_levels'].get(imp, 0) + 1
        
        # Critic prediction statistics
        if example.critic_predictions:
            stats['critic_predictions']['total_with_predictions'] += 1
            
            if example.critic_predictions.has_errors:
                stats['critic_predictions']['critic_says_errors'] += 1
            else:
                stats['critic_predictions']['critic_says_clean'] += 1
            
            # Calculate accuracy (compare with ground truth)
            ground_truth = example.error_injection.has_errors
            prediction = example.critic_predictions.has_errors
            stats['critic_predictions']['accuracy']['total_comparable'] += 1
            
            if ground_truth == prediction:
                stats['critic_predictions']['accuracy']['correct'] += 1
            else:
                stats['critic_predictions']['accuracy']['incorrect'] += 1
    
    # Calculate average steps
    stats['step_counts']['avg'] = total_steps / len(examples) if examples else 0
    if stats['step_counts']['min'] == float('inf'):
        stats['step_counts']['min'] = 0
    
    return stats