#!/usr/bin/env python3
"""
Check the progress of the large-scale error injection experiment.
"""

import json
import os
from datetime import datetime

def check_latest_experiment():
    """Check progress of the latest experiment."""
    
    experiments_dir = "data/experiments"
    experiment_dirs = [d for d in os.listdir(experiments_dir) if d.startswith('large_scale_')]
    
    if not experiment_dirs:
        print("❌ No experiment directories found.")
        return
    
    # Use the most recent experiment
    latest_experiment = sorted(experiment_dirs)[-1]
    experiment_path = f"{experiments_dir}/{latest_experiment}"
    
    print(f"📊 Experiment Progress: {latest_experiment}")
    print("=" * 60)
    
    # Check experiment metadata
    metadata_file = f"{experiment_path}/experiment_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"🎯 Target: {metadata['total_problems']} problems")
        print(f"📅 Started: {metadata['created_timestamp']}")
        print(f"🔧 Settings: {metadata['error_placement']}, min {metadata['min_steps']} steps")
        print()
    
    # Check latest checkpoint
    checkpoints_dir = f"{experiment_path}/checkpoints"
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.startswith('checkpoint_batch_')]
        
        if checkpoint_files:
            # Get latest checkpoint
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            
            checkpoint_path = f"{checkpoints_dir}/{latest_checkpoint}"
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            total_processed = checkpoint_data['total_processed']
            batch_num = checkpoint_data['batch_number']
            timestamp = checkpoint_data['timestamp']
            
            # Calculate statistics
            results = checkpoint_data['results']
            successful = sum(1 for r in results if r['success'])
            success_rate = successful / len(results) if results else 0
            
            print(f"📈 Progress:")
            print(f"   Processed: {total_processed}/500 ({100*total_processed/500:.1f}%)")
            print(f"   Batches completed: {batch_num}")
            print(f"   Success rate: {success_rate:.1%} ({successful}/{len(results)})")
            print(f"   Last update: {timestamp}")
            
            # Error type distribution
            error_types = {}
            for result in results:
                if result['success']:
                    error_type = result['error_analysis'].get('error_type', 'unknown')
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if error_types:
                print(f"\\n🎲 Error Type Distribution:")
                for error_type, count in sorted(error_types.items()):
                    percentage = 100 * count / successful
                    print(f"   {error_type}: {count} ({percentage:.1f}%)")
            
            # Estimate completion time
            if total_processed > 0:
                remaining = 500 - total_processed
                # Parse timestamp to calculate time taken
                start_time = datetime.fromisoformat(metadata['created_timestamp'].replace('Z', '+00:00'))
                current_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                elapsed = (current_time - start_time).total_seconds()
                
                rate = total_processed / elapsed  # problems per second
                estimated_remaining = remaining / rate if rate > 0 else 0
                
                print(f"\\n⏱️  Timing:")
                print(f"   Elapsed: {elapsed/60:.1f} minutes")
                print(f"   Rate: {rate*60:.1f} problems/minute")
                if estimated_remaining > 0:
                    print(f"   ETA: {estimated_remaining/60:.1f} minutes remaining")
            
        else:
            print("📋 No checkpoints found - experiment may not have started yet.")
    
    # Check if there are final results
    results_dir = f"{experiment_path}/results"
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.startswith('final_results_')]
        
        if result_files:
            print(f"\\n✅ EXPERIMENT COMPLETED!")
            latest_result = sorted(result_files)[-1]
            print(f"   Final results: {latest_result}")
            
            # Check if summary exists
            summary_file = f"{results_dir}/experiment_summary.json"
            if os.path.exists(summary_file):
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                print(f"\\n📊 Final Summary:")
                print(f"   Total problems: {summary['total_problems']}")
                print(f"   Successful: {summary['successful_injections']}")
                print(f"   Success rate: {summary['success_rate']:.1%}")
                print(f"   Avg error placement: {summary['average_error_placement_percent']:.1f}%")
                print(f"   Errors in last third: {summary['placement_in_last_third']}")

if __name__ == "__main__":
    check_latest_experiment()