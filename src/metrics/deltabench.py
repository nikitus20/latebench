"""
DeltaBench-Compatible Metrics for Mathematical Reasoning Evaluation

This module implements research-grade evaluation metrics compatible with DeltaBench
standards, including first-error cutoff logic and comprehensive performance analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import logging

from src.critic import CriticResult
from src.data_processing.unified_schema import LateBenchExample

logger = logging.getLogger(__name__)


@dataclass
class StepPrediction:
    """Individual step prediction with confidence"""
    step_number: int
    is_error: bool
    confidence: float = 1.0
    explanation: str = ""


@dataclass
class ErrorBoundary:
    """Defines error boundaries for evaluation"""
    first_error_step: Optional[int]
    total_steps: int
    valid_steps: List[int]  # Steps to consider for evaluation
    
    def is_valid_prediction(self, step_number: int) -> bool:
        """Check if prediction for this step should be evaluated"""
        return step_number in self.valid_steps


@dataclass
class DeltaBenchMetrics:
    """Comprehensive evaluation metrics following DeltaBench standards"""
    
    # Step-level metrics (micro-averaged)
    step_precision: float
    step_recall: float
    step_f1: float
    
    # Example-level metrics (macro-averaged)
    example_precision: float
    example_recall: float
    example_f1: float
    
    # Error detection metrics
    error_detection_accuracy: float  # Correctly identified if example has errors
    first_error_accuracy: float      # Correctly identified first error step
    false_positive_rate: float       # Predicted errors where none exist
    
    # Advanced metrics
    early_detection_rate: float      # Detected errors before or at first error
    late_detection_rate: float       # Detected errors after first error
    step_accuracy: float            # Overall step-level classification accuracy
    
    # Confidence metrics
    calibration_error: float        # Confidence calibration quality
    auc_roc: float                 # Area under ROC curve
    
    # Detailed breakdowns
    confusion_matrix: Dict[str, int]
    per_example_metrics: List[Dict[str, Any]]
    error_type_breakdown: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            'step_level': {
                'precision': self.step_precision,
                'recall': self.step_recall,
                'f1': self.step_f1
            },
            'example_level': {
                'precision': self.example_precision,
                'recall': self.example_recall,
                'f1': self.example_f1
            },
            'error_detection': {
                'accuracy': self.error_detection_accuracy,
                'first_error_accuracy': self.first_error_accuracy,
                'false_positive_rate': self.false_positive_rate
            },
            'advanced': {
                'early_detection_rate': self.early_detection_rate,
                'late_detection_rate': self.late_detection_rate,
                'step_accuracy': self.step_accuracy,
                'calibration_error': self.calibration_error,
                'auc_roc': self.auc_roc
            },
            'confusion_matrix': self.confusion_matrix,
            'per_example_metrics': self.per_example_metrics,
            'error_type_breakdown': self.error_type_breakdown
        }


class DeltaBenchEvaluator:
    """Evaluator implementing DeltaBench-compatible metrics"""
    
    def __init__(self, use_first_error_cutoff: bool = True):
        self.use_first_error_cutoff = use_first_error_cutoff
        
    def extract_ground_truth(self, example: LateBenchExample) -> Tuple[List[StepPrediction], ErrorBoundary]:
        """Extract ground truth labels from LateBench example"""
        
        ground_truth_steps = []
        error_steps = []
        
        # Extract step-level ground truth
        for step in example.solution.steps:
            is_error = step.is_error
            if is_error:
                error_steps.append(step.step_number)
            
            ground_truth_steps.append(StepPrediction(
                step_number=step.step_number,
                is_error=is_error,
                confidence=1.0
            ))
        
        # Determine first error step
        first_error_step = min(error_steps) if error_steps else None
        
        # Define evaluation boundary
        if self.use_first_error_cutoff and first_error_step is not None:
            # Only evaluate steps up to and including first error
            valid_steps = [s.step_number for s in example.solution.steps 
                          if s.step_number <= first_error_step]
        else:
            # Evaluate all steps
            valid_steps = [s.step_number for s in example.solution.steps]
        
        boundary = ErrorBoundary(
            first_error_step=first_error_step,
            total_steps=len(example.solution.steps),
            valid_steps=valid_steps
        )
        
        return ground_truth_steps, boundary
    
    def extract_predictions(self, critic_result: CriticResult, 
                          boundary: ErrorBoundary) -> List[StepPrediction]:
        """Extract predictions from critic result within evaluation boundary"""
        
        predictions = []
        predicted_error_steps = set(critic_result.error_steps)
        
        # Create prediction for each valid step
        for step_num in boundary.valid_steps:
            is_error = step_num in predicted_error_steps
            explanation = critic_result.explanations.get(step_num, "")
            
            predictions.append(StepPrediction(
                step_number=step_num,
                is_error=is_error,
                confidence=1.0,  # Could be enhanced with confidence extraction
                explanation=explanation
            ))
        
        return predictions
    
    def compute_step_level_metrics(self, ground_truth: List[StepPrediction],
                                 predictions: List[StepPrediction]) -> Dict[str, float]:
        """Compute micro-averaged step-level metrics"""
        
        # Align predictions with ground truth
        gt_dict = {pred.step_number: pred.is_error for pred in ground_truth}
        pred_dict = {pred.step_number: pred.is_error for pred in predictions}
        
        # Get common steps
        common_steps = set(gt_dict.keys()) & set(pred_dict.keys())
        
        if not common_steps:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0}
        
        # Calculate confusion matrix elements
        tp = sum(1 for step in common_steps 
                if gt_dict[step] and pred_dict[step])
        fp = sum(1 for step in common_steps 
                if not gt_dict[step] and pred_dict[step])
        fn = sum(1 for step in common_steps 
                if gt_dict[step] and not pred_dict[step])
        tn = sum(1 for step in common_steps 
                if not gt_dict[step] and not pred_dict[step])
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(common_steps)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def compute_example_level_metrics(self, ground_truth: List[StepPrediction],
                                    predictions: List[StepPrediction],
                                    boundary: ErrorBoundary) -> Dict[str, Any]:
        """Compute example-level metrics"""
        
        # Check if example has errors (in valid range)
        gt_has_errors = any(pred.is_error for pred in ground_truth 
                           if boundary.is_valid_prediction(pred.step_number))
        pred_has_errors = any(pred.is_error for pred in predictions
                             if boundary.is_valid_prediction(pred.step_number))
        
        # Error detection accuracy
        error_detection_correct = (gt_has_errors == pred_has_errors)
        
        # First error accuracy
        first_error_correct = False
        if boundary.first_error_step is not None:
            predicted_first_error = None
            for pred in predictions:
                if pred.is_error and boundary.is_valid_prediction(pred.step_number):
                    predicted_first_error = pred.step_number
                    break
            first_error_correct = (predicted_first_error == boundary.first_error_step)
        
        # Early/late detection
        early_detection = False
        late_detection = False
        if boundary.first_error_step is not None and pred_has_errors:
            predicted_error_steps = [pred.step_number for pred in predictions 
                                   if pred.is_error and boundary.is_valid_prediction(pred.step_number)]
            if predicted_error_steps:
                earliest_prediction = min(predicted_error_steps)
                latest_prediction = max(predicted_error_steps)
                
                early_detection = earliest_prediction <= boundary.first_error_step
                late_detection = latest_prediction > boundary.first_error_step
        
        return {
            'has_errors_gt': gt_has_errors,
            'has_errors_pred': pred_has_errors,
            'error_detection_correct': error_detection_correct,
            'first_error_correct': first_error_correct,
            'early_detection': early_detection,
            'late_detection': late_detection,
            'first_error_step_gt': boundary.first_error_step,
            'predicted_error_steps': [pred.step_number for pred in predictions if pred.is_error]
        }
    
    def evaluate_batch(self, examples: List[LateBenchExample],
                      critic_results: Dict[str, CriticResult]) -> DeltaBenchMetrics:
        """Evaluate batch of examples with comprehensive metrics"""
        
        logger.info(f"Computing DeltaBench metrics for {len(examples)} examples")
        
        # Collect all step-level and example-level metrics
        all_step_metrics = []
        all_example_metrics = []
        per_example_details = []
        
        # Process each example
        for example in examples:
            if example.id not in critic_results:
                logger.warning(f"No critic result for example {example.id}")
                continue
            
            critic_result = critic_results[example.id]
            if not critic_result:
                logger.warning(f"Invalid critic result for example {example.id}")
                continue
            
            # Extract ground truth and predictions
            ground_truth, boundary = self.extract_ground_truth(example)
            predictions = self.extract_predictions(critic_result, boundary)
            
            # Compute metrics for this example
            step_metrics = self.compute_step_level_metrics(ground_truth, predictions)
            example_metrics = self.compute_example_level_metrics(ground_truth, predictions, boundary)
            
            all_step_metrics.append(step_metrics)
            all_example_metrics.append(example_metrics)
            
            # Store detailed results
            per_example_details.append({
                'example_id': example.id,
                'step_metrics': step_metrics,
                'example_metrics': example_metrics,
                'boundary': {
                    'first_error_step': boundary.first_error_step,
                    'total_steps': boundary.total_steps,
                    'valid_steps': boundary.valid_steps
                }
            })
        
        # Aggregate metrics
        return self._aggregate_metrics(all_step_metrics, all_example_metrics, per_example_details)
    
    def _aggregate_metrics(self, step_metrics: List[Dict[str, float]],
                         example_metrics: List[Dict[str, Any]],
                         per_example_details: List[Dict[str, Any]]) -> DeltaBenchMetrics:
        """Aggregate individual metrics into final DeltaBench metrics"""
        
        if not step_metrics or not example_metrics:
            return self._empty_metrics()
        
        # Micro-averaged step-level metrics (sum all TP, FP, FN, TN)
        total_tp = sum(m['tp'] for m in step_metrics)
        total_fp = sum(m['fp'] for m in step_metrics)
        total_fn = sum(m['fn'] for m in step_metrics)
        total_tn = sum(m['tn'] for m in step_metrics)
        
        step_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        step_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        step_f1 = 2 * step_precision * step_recall / (step_precision + step_recall) if (step_precision + step_recall) > 0 else 0.0
        step_accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
        
        # Macro-averaged example-level metrics
        example_precisions = []
        example_recalls = []
        
        for detail in per_example_details:
            ex_metrics = detail['step_metrics']
            if ex_metrics['tp'] + ex_metrics['fp'] > 0:
                example_precisions.append(ex_metrics['precision'])
            if ex_metrics['tp'] + ex_metrics['fn'] > 0:
                example_recalls.append(ex_metrics['recall'])
        
        example_precision = np.mean(example_precisions) if example_precisions else 0.0
        example_recall = np.mean(example_recalls) if example_recalls else 0.0
        example_f1 = 2 * example_precision * example_recall / (example_precision + example_recall) if (example_precision + example_recall) > 0 else 0.0
        
        # Error detection metrics
        error_detection_correct = sum(1 for m in example_metrics if m['error_detection_correct'])
        error_detection_accuracy = error_detection_correct / len(example_metrics)
        
        first_error_correct = sum(1 for m in example_metrics if m['first_error_correct'])
        first_error_accuracy = first_error_correct / len(example_metrics)
        
        false_positives = sum(1 for m in example_metrics if not m['has_errors_gt'] and m['has_errors_pred'])
        false_positive_rate = false_positives / len(example_metrics)
        
        # Advanced metrics
        early_detections = sum(1 for m in example_metrics if m['early_detection'])
        late_detections = sum(1 for m in example_metrics if m['late_detection'])
        examples_with_errors = sum(1 for m in example_metrics if m['has_errors_gt'])
        
        early_detection_rate = early_detections / examples_with_errors if examples_with_errors > 0 else 0.0
        late_detection_rate = late_detections / examples_with_errors if examples_with_errors > 0 else 0.0
        
        # Simplified confidence metrics (would need actual confidence scores for full implementation)
        calibration_error = 0.0  # Placeholder
        auc_roc = 0.5  # Placeholder
        
        # Confusion matrix
        confusion_matrix = {
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn,
            'tn': total_tn
        }
        
        # Error type breakdown (simplified)
        error_type_breakdown = {
            'natural_errors': {
                'precision': step_precision,
                'recall': step_recall,
                'f1': step_f1
            }
        }
        
        return DeltaBenchMetrics(
            step_precision=step_precision,
            step_recall=step_recall,
            step_f1=step_f1,
            example_precision=example_precision,
            example_recall=example_recall,
            example_f1=example_f1,
            error_detection_accuracy=error_detection_accuracy,
            first_error_accuracy=first_error_accuracy,
            false_positive_rate=false_positive_rate,
            early_detection_rate=early_detection_rate,
            late_detection_rate=late_detection_rate,
            step_accuracy=step_accuracy,
            calibration_error=calibration_error,
            auc_roc=auc_roc,
            confusion_matrix=confusion_matrix,
            per_example_metrics=per_example_details,
            error_type_breakdown=error_type_breakdown
        )
    
    def _empty_metrics(self) -> DeltaBenchMetrics:
        """Return empty metrics for edge cases"""
        return DeltaBenchMetrics(
            step_precision=0.0,
            step_recall=0.0,
            step_f1=0.0,
            example_precision=0.0,
            example_recall=0.0,
            example_f1=0.0,
            error_detection_accuracy=0.0,
            first_error_accuracy=0.0,
            false_positive_rate=0.0,
            early_detection_rate=0.0,
            late_detection_rate=0.0,
            step_accuracy=0.0,
            calibration_error=0.0,
            auc_roc=0.5,
            confusion_matrix={'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0},
            per_example_metrics=[],
            error_type_breakdown={}
        )


# Convenience functions
def evaluate_with_deltabench_metrics(examples: List[LateBenchExample],
                                   critic_results: Dict[str, CriticResult],
                                   use_first_error_cutoff: bool = True) -> DeltaBenchMetrics:
    """Convenience function for DeltaBench evaluation"""
    evaluator = DeltaBenchEvaluator(use_first_error_cutoff=use_first_error_cutoff)
    return evaluator.evaluate_batch(examples, critic_results)


def print_metrics_summary(metrics: DeltaBenchMetrics):
    """Print human-readable metrics summary"""
    print("=" * 60)
    print("DeltaBench Metrics Summary")
    print("=" * 60)
    
    print(f"\nStep-Level Metrics (Micro-averaged):")
    print(f"  Precision: {metrics.step_precision:.3f}")
    print(f"  Recall:    {metrics.step_recall:.3f}")
    print(f"  F1-Score:  {metrics.step_f1:.3f}")
    print(f"  Accuracy:  {metrics.step_accuracy:.3f}")
    
    print(f"\nExample-Level Metrics (Macro-averaged):")
    print(f"  Precision: {metrics.example_precision:.3f}")
    print(f"  Recall:    {metrics.example_recall:.3f}")
    print(f"  F1-Score:  {metrics.example_f1:.3f}")
    
    print(f"\nError Detection:")
    print(f"  Detection Accuracy:   {metrics.error_detection_accuracy:.3f}")
    print(f"  First Error Accuracy: {metrics.first_error_accuracy:.3f}")
    print(f"  False Positive Rate:  {metrics.false_positive_rate:.3f}")
    
    print(f"\nAdvanced Metrics:")
    print(f"  Early Detection Rate: {metrics.early_detection_rate:.3f}")
    print(f"  Late Detection Rate:  {metrics.late_detection_rate:.3f}")
    
    print(f"\nConfusion Matrix:")
    cm = metrics.confusion_matrix
    print(f"  TP: {cm['tp']}, FP: {cm['fp']}")
    print(f"  FN: {cm['fn']}, TN: {cm['tn']}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage would go here
    pass