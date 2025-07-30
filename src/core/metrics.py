"""
Unified metrics system for LateBench evaluation.
Core functionality: DeltaBench-compatible step-level metrics using LateBenchExample objects.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
from unified_schema import LateBenchExample


@dataclass
class DeltaBenchMetrics:
    """DeltaBench evaluation metrics."""
    # Step-level metrics (micro-averaged)
    step_precision: float
    step_recall: float
    step_f1: float
    step_accuracy: float
    
    # Example-level metrics (macro-averaged)
    example_precision: float
    example_recall: float
    example_f1: float
    
    # Error detection metrics
    error_detection_accuracy: float
    first_error_accuracy: float
    false_positive_rate: float
    early_detection_rate: float
    
    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Metadata
    total_examples: int
    total_steps: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'step_precision': self.step_precision,
            'step_recall': self.step_recall,
            'step_f1': self.step_f1,
            'step_accuracy': self.step_accuracy,
            'example_precision': self.example_precision,
            'example_recall': self.example_recall,
            'example_f1': self.example_f1,
            'error_detection_accuracy': self.error_detection_accuracy,
            'first_error_accuracy': self.first_error_accuracy,
            'false_positive_rate': self.false_positive_rate,
            'early_detection_rate': self.early_detection_rate,
            'confusion_matrix': {
                'true_positives': self.true_positives,
                'false_positives': self.false_positives,
                'true_negatives': self.true_negatives,
                'false_negatives': self.false_negatives
            },
            'total_examples': self.total_examples,
            'total_steps': self.total_steps
        }


class DeltaBenchEvaluator:
    """Evaluate critic performance using DeltaBench metrics."""
    
    def __init__(self, use_first_error_cutoff: bool = True):
        self.use_first_error_cutoff = use_first_error_cutoff

    def evaluate_batch(self, examples: List[LateBenchExample], 
                      evaluation_mode: str = "auto") -> DeltaBenchMetrics:
        """Evaluate a batch of LateBenchExample objects using embedded critic predictions.
        
        Args:
            examples: List of LateBenchExample objects with critic predictions
            evaluation_mode: "original", "injected", or "auto" - which predictions to evaluate
        """
        
        step_metrics = []
        example_metrics = []
        
        total_tp = total_fp = total_tn = total_fn = 0
        correct_detections = 0
        correct_first_errors = 0
        early_detections = 0
        
        for example in examples:
            # Get ground truth error steps using explicit fields
            ground_truth_steps = self._get_ground_truth_error_steps_explicit(example, evaluation_mode)
            
            # Get critic predictions from embedded fields
            critic_prediction = self._get_critic_prediction(example, evaluation_mode)
            if not critic_prediction:
                continue  # Skip examples without critic predictions
            
            predicted_steps = critic_prediction.error_steps if critic_prediction.error_steps else []
            
            # Apply first-error cutoff if enabled
            if self.use_first_error_cutoff and ground_truth_steps:
                first_error_step = min(ground_truth_steps)
                predicted_steps = [step for step in predicted_steps if step <= first_error_step]
            
            # Calculate step-level metrics for this example
            total_solution_steps = len(example.solution.steps)
            
            if total_solution_steps == 0:
                continue
            
            # Calculate confusion matrix for this example
            tp = len(set(predicted_steps) & set(ground_truth_steps))
            fp = len(set(predicted_steps) - set(ground_truth_steps))
            fn = len(set(ground_truth_steps) - set(predicted_steps))
            tn = total_solution_steps - tp - fp - fn
            
            # Ensure non-negative values
            tn = max(0, tn)
            
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            
            # Step-level precision/recall for this example
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            step_metrics.append({'precision': precision, 'recall': recall, 'f1': f1})
            
            # Example-level metrics
            has_ground_truth_errors = len(ground_truth_steps) > 0
            has_predicted_errors = len(predicted_steps) > 0
            
            if has_ground_truth_errors and has_predicted_errors:
                correct_detections += 1
                example_metrics.append({'precision': 1.0, 'recall': 1.0})
                
                # Check if first error was caught
                if ground_truth_steps and predicted_steps:
                    first_gt_error = min(ground_truth_steps)
                    if first_gt_error in predicted_steps:
                        correct_first_errors += 1
                    
                    # Check early detection (error found at or before actual error)
                    if any(pred <= first_gt_error for pred in predicted_steps):
                        early_detections += 1
                        
            elif has_ground_truth_errors and not has_predicted_errors:
                example_metrics.append({'precision': 0.0, 'recall': 0.0})
            elif not has_ground_truth_errors and has_predicted_errors:
                example_metrics.append({'precision': 0.0, 'recall': 1.0})
            else:
                correct_detections += 1
                example_metrics.append({'precision': 1.0, 'recall': 1.0})
        
        # Calculate overall metrics
        total_examples = len(step_metrics)
        total_steps = total_tp + total_fp + total_tn + total_fn
        
        if total_examples == 0:
            return self._create_empty_metrics()
        
        # Step-level metrics (micro-averaged)
        step_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        step_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        step_f1 = 2 * step_precision * step_recall / (step_precision + step_recall) if (step_precision + step_recall) > 0 else 0.0
        step_accuracy = (total_tp + total_tn) / total_steps if total_steps > 0 else 0.0
        
        # Example-level metrics (macro-averaged)
        avg_example_precision = sum(m['precision'] for m in example_metrics) / len(example_metrics)
        avg_example_recall = sum(m['recall'] for m in example_metrics) / len(example_metrics)
        avg_example_f1 = 2 * avg_example_precision * avg_example_recall / (avg_example_precision + avg_example_recall) if (avg_example_precision + avg_example_recall) > 0 else 0.0
        
        # Error detection metrics
        error_detection_accuracy = correct_detections / total_examples
        first_error_accuracy = correct_first_errors / total_examples
        false_positive_rate = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0.0
        early_detection_rate = early_detections / total_examples
        
        return DeltaBenchMetrics(
            step_precision=step_precision,
            step_recall=step_recall,
            step_f1=step_f1,
            step_accuracy=step_accuracy,
            example_precision=avg_example_precision,
            example_recall=avg_example_recall,
            example_f1=avg_example_f1,
            error_detection_accuracy=error_detection_accuracy,
            first_error_accuracy=first_error_accuracy,
            false_positive_rate=false_positive_rate,
            early_detection_rate=early_detection_rate,
            true_positives=total_tp,
            false_positives=total_fp,
            true_negatives=total_tn,
            false_negatives=total_fn,
            total_examples=total_examples,
            total_steps=total_steps
        )

    def _get_ground_truth_error_steps_explicit(self, example: LateBenchExample, evaluation_mode: str) -> List[int]:
        """Get ground truth error steps from explicit fields - no fallback logic."""
        if evaluation_mode == "original":
            return example.original_error_steps  # Could be [] if no errors
        elif evaluation_mode == "injected":
            return example.injected_error_steps  # Could be [] if no injection
        elif evaluation_mode == "auto":
            # Use injected if available, otherwise original
            return example.injected_error_steps if example.injected_error_steps else example.original_error_steps
        else:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")
    
    def _get_critic_prediction(self, example: LateBenchExample, evaluation_mode: str):
        """Get critic prediction from embedded fields based on evaluation mode."""
        if evaluation_mode == "original":
            return example.critic_predictions_original
        elif evaluation_mode == "injected":
            return example.critic_predictions_injected
        elif evaluation_mode == "auto":
            # Use injected if available, otherwise original
            return example.critic_predictions_injected if example.critic_predictions_injected else example.critic_predictions_original
        else:
            raise ValueError(f"Invalid evaluation_mode: {evaluation_mode}")

    def _create_empty_metrics(self) -> DeltaBenchMetrics:
        """Create empty metrics for edge cases."""
        return DeltaBenchMetrics(
            step_precision=0.0,
            step_recall=0.0,
            step_f1=0.0,
            step_accuracy=0.0,
            example_precision=0.0,
            example_recall=0.0,
            example_f1=0.0,
            error_detection_accuracy=0.0,
            first_error_accuracy=0.0,
            false_positive_rate=0.0,
            early_detection_rate=0.0,
            true_positives=0,
            false_positives=0,
            true_negatives=0,
            false_negatives=0,
            total_examples=0,
            total_steps=0
        )


def print_metrics_summary(metrics: DeltaBenchMetrics):
    """Print formatted metrics summary."""
    print("\n" + "="*70)
    print("DELTABENCH EVALUATION METRICS")
    print("="*70)
    
    print(f"\n📊 STEP-LEVEL METRICS (Micro-averaged)")
    print(f"   Precision: {metrics.step_precision:.3f}")
    print(f"   Recall:    {metrics.step_recall:.3f}")
    print(f"   F1-Score:  {metrics.step_f1:.3f}")
    print(f"   Accuracy:  {metrics.step_accuracy:.3f}")
    
    print(f"\n📈 EXAMPLE-LEVEL METRICS (Macro-averaged)")
    print(f"   Precision: {metrics.example_precision:.3f}")
    print(f"   Recall:    {metrics.example_recall:.3f}")
    print(f"   F1-Score:  {metrics.example_f1:.3f}")
    
    print(f"\n🎯 ERROR DETECTION METRICS")
    print(f"   Detection Accuracy:  {metrics.error_detection_accuracy:.3f}")
    print(f"   First Error Acc:     {metrics.first_error_accuracy:.3f}")
    print(f"   False Positive Rate: {metrics.false_positive_rate:.3f}")
    print(f"   Early Detection:     {metrics.early_detection_rate:.3f}")
    
    print(f"\n📋 CONFUSION MATRIX")
    print(f"   True Positives:  {metrics.true_positives}")
    print(f"   False Positives: {metrics.false_positives}")
    print(f"   True Negatives:  {metrics.true_negatives}")
    print(f"   False Negatives: {metrics.false_negatives}")
    
    print(f"\n📊 DATASET STATS")
    print(f"   Total Examples: {metrics.total_examples}")
    print(f"   Total Steps:    {metrics.total_steps}")
    
    print("="*70)


def evaluate_critic_on_dataset(examples: List[LateBenchExample], evaluation_mode: str = "auto", 
                              use_first_error_cutoff: bool = True) -> DeltaBenchMetrics:
    """Convenience function to evaluate critic performance on a dataset.
    
    Args:
        examples: List of LateBenchExample objects with critic predictions
        evaluation_mode: "original", "injected", or "auto" - which predictions to evaluate
        use_first_error_cutoff: Whether to apply first-error cutoff for evaluation
    
    Returns:
        DeltaBenchMetrics object with evaluation results
    """
    evaluator = DeltaBenchEvaluator(use_first_error_cutoff=use_first_error_cutoff)
    return evaluator.evaluate_batch(examples, evaluation_mode=evaluation_mode)