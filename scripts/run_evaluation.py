#!/usr/bin/env python3
"""
Clean evaluation runner that manages results storage properly
"""

import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_processing.unified_schema import LateBenchExample
from core.metrics import evaluate_critic_on_dataset, print_metrics_summary
from utils.storage import load_examples_from_file

def create_results_structure():
    """Ensure results directory structure exists"""
    Path("results/experiments").mkdir(parents=True, exist_ok=True)
    Path("results/evaluations").mkdir(parents=True, exist_ok=True)
    Path("results/metrics").mkdir(parents=True, exist_ok=True)

def generate_experiment_id(dataset: str, operation: str, details: str = None) -> str:
    """Generate standardized experiment ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if details:
        return f"{dataset}_{operation}_{details}_{timestamp}"
    return f"{dataset}_{operation}_{timestamp}"

def compute_metrics_on_results(experiment_file: str, output_name: str = None):
    """Compute metrics on experiment results"""
    
    print(f"📊 Computing Metrics on: {experiment_file}")
    print("=" * 60)
    
    # Load results
    with open(experiment_file, 'r') as f:
        data = json.load(f)
    
    # Convert to LateBenchExample objects
    examples = [LateBenchExample.from_dict(ex) for ex in data]
    
    # Filter to examples with predictions
    examples_with_predictions = [
        ex for ex in examples 
        if ex.critic_predictions_original is not None or ex.critic_predictions_injected is not None
    ]
    
    print(f"✅ Loaded {len(examples)} examples, {len(examples_with_predictions)} with predictions")
    
    if not examples_with_predictions:
        print("❌ No examples have predictions!")
        return None
    
    # Determine evaluation mode based on what predictions exist
    if examples_with_predictions[0].critic_predictions_original:
        eval_mode = "original"
    elif examples_with_predictions[0].critic_predictions_injected:
        eval_mode = "injected"
    else:
        eval_mode = "auto"
    
    print(f"🔍 Using evaluation mode: {eval_mode}")
    
    # Compute metrics
    metrics = evaluate_critic_on_dataset(
        examples_with_predictions, 
        evaluation_mode=eval_mode,
        use_first_error_cutoff=True
    )
    
    # Display results
    print_metrics_summary(metrics)
    
    # Save evaluation results
    if not output_name:
        experiment_path = Path(experiment_file)
        output_name = f"{experiment_path.stem}_metrics"
    
    evaluation_file = f"results/evaluations/{output_name}.json"
    
    evaluation_results = {
        "experiment_file": experiment_file,
        "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
        "evaluation_mode": eval_mode,
        "total_examples": len(examples_with_predictions),
        "metrics": metrics.to_dict()
    }
    
    with open(evaluation_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"\n💾 Evaluation results saved to: {evaluation_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="LateBench Evaluation Runner")
    parser.add_argument("experiment_file", help="Path to experiment results file")
    parser.add_argument("--output-name", help="Name for evaluation output (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Ensure results structure exists
    create_results_structure()
    
    # Check if file exists
    if not Path(args.experiment_file).exists():
        print(f"❌ Experiment file not found: {args.experiment_file}")
        return 1
    
    # Compute metrics
    try:
        metrics = compute_metrics_on_results(args.experiment_file, args.output_name)
        if metrics:
            print("\n🎯 Evaluation completed successfully!")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())