"""
Simplified LateBench Dashboard for Error Injection Review.
Core functionality: load problems, inject errors, manually review, make decisions.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, request, jsonify
import json
import time
from typing import Dict, Any, Optional, List

from core.data_loader import LateBenchDataLoader
from core.error_injector import ErrorInjector
from core.critic import MathCritic, evaluate_single_example
from utils.storage import LateBenchStorage

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'latebench_simple_dashboard'

# Global components
data_loader = LateBenchDataLoader()
error_injector = ErrorInjector()
critic = MathCritic()
storage = LateBenchStorage()

# Current session state
current_dataset = None
current_examples = []
current_example_index = 0
manual_decisions = {}  # example_id -> {"decision": "yes/maybe/no", "notes": "..."}


@app.route('/')
def index():
    """Main dashboard page."""
    datasets = data_loader.list_available_datasets()
    
    context = {
        'datasets': datasets,
        'current_dataset': current_dataset,
        'current_example': get_current_example(),
        'total_examples': len(current_examples),
        'current_index': current_example_index,
        'manual_decisions': manual_decisions
    }
    
    return render_template('simple_dashboard.html', **context)


@app.route('/api/load_dataset', methods=['POST'])
def load_dataset():
    """Load a dataset for review."""
    global current_dataset, current_examples, current_example_index
    
    data = request.get_json()
    dataset_name = data.get('dataset_name')
    problem_type = data.get('problem_type', 'all')
    
    try:
        examples = data_loader.load_dataset(dataset_name, problem_type)
        
        current_dataset = f"{dataset_name}_{problem_type}"
        current_examples = examples
        current_example_index = 0
        
        return jsonify({
            'success': True,
            'dataset': current_dataset,
            'total_examples': len(examples),
            'current_example': get_current_example()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/navigate/<direction>')
def navigate(direction):
    """Navigate between examples."""
    global current_example_index
    
    if direction == 'next':
        current_example_index = min(current_example_index + 1, len(current_examples) - 1)
    elif direction == 'prev':
        current_example_index = max(current_example_index - 1, 0)
    elif direction == 'first':
        current_example_index = 0
    elif direction == 'last':
        current_example_index = len(current_examples) - 1
    
    return jsonify({
        'success': True,
        'current_index': current_example_index,
        'current_example': get_current_example()
    })


@app.route('/api/inject_error', methods=['POST'])
def inject_error():
    """Inject error into current example."""
    current_example = get_current_example()
    if not current_example:
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    custom_suggestion = data.get('custom_suggestion')
    
    try:
        # Convert to injection format
        injection_problem = convert_example_for_injection(current_example)
        
        # Run error injection
        result = error_injector.inject_error(injection_problem, custom_suggestion)
        
        if result.success:
            return jsonify({
                'success': True,
                'injection_result': {
                    'modified_solution': result.modified_solution,
                    'error_analysis': result.error_analysis,
                    'error_explanation': result.error_explanation,
                    'metadata': result.metadata
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result.error_message
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/evaluate_with_critic', methods=['POST'])
def evaluate_with_critic():
    """Evaluate current example with critic."""
    current_example = get_current_example()
    if not current_example:
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    modified_solution = data.get('modified_solution')
    
    try:
        # Prepare for critic evaluation
        problem = current_example.get('problem', {}).get('statement', '')
        
        if modified_solution:
            # Use modified solution from error injection
            steps = modified_solution.get('steps', [])
        else:
            # Use original solution
            steps = current_example.get('solution', {}).get('steps', [])
        
        # Run critic evaluation
        critic_result = critic.evaluate_solution(problem, steps)
        
        return jsonify({
            'success': True,
            'critic_result': critic_result.to_dict()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/save_decision', methods=['POST'])
def save_decision():
    """Save manual decision for current example."""
    current_example = get_current_example()
    if not current_example:
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    decision = data.get('decision')  # 'yes', 'maybe', 'no'
    notes = data.get('notes', '')
    
    if decision not in ['yes', 'maybe', 'no']:
        return jsonify({'success': False, 'error': 'Invalid decision'}), 400
    
    try:
        example_id = current_example.get('id', f'example_{current_example_index}')
        
        manual_decisions[example_id] = {
            'decision': decision,
            'notes': notes,
            'timestamp': str(int(time.time())),
            'example_index': current_example_index
        }
        
        # Save to persistent storage
        save_manual_decisions()
        
        return jsonify({
            'success': True,
            'decision': decision,
            'example_id': example_id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export_decisions')
def export_decisions():
    """Export manual decisions."""
    try:
        # Prepare export data
        export_data = {
            'dataset': current_dataset,
            'total_examples': len(current_examples),
            'total_decisions': len(manual_decisions),
            'decisions_by_type': {
                'yes': sum(1 for d in manual_decisions.values() if d['decision'] == 'yes'),
                'maybe': sum(1 for d in manual_decisions.values() if d['decision'] == 'maybe'),
                'no': sum(1 for d in manual_decisions.values() if d['decision'] == 'no')
            },
            'decisions': manual_decisions
        }
        
        # Save export
        output_path = storage.save_results(
            [export_data], 
            f"manual_decisions_{current_dataset}",
            {'export_type': 'manual_decisions', 'dataset': current_dataset}
        )
        
        return jsonify({
            'success': True,
            'export_path': output_path,
            'stats': export_data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats')
def get_stats():
    """Get current session statistics."""
    stats = {
        'dataset': current_dataset,
        'total_examples': len(current_examples),
        'current_index': current_example_index,
        'decisions_made': len(manual_decisions),
        'decisions_by_type': {
            'yes': sum(1 for d in manual_decisions.values() if d['decision'] == 'yes'),
            'maybe': sum(1 for d in manual_decisions.values() if d['decision'] == 'maybe'),
            'no': sum(1 for d in manual_decisions.values() if d['decision'] == 'no')
        },
        'completion_rate': len(manual_decisions) / len(current_examples) if current_examples else 0
    }
    
    return jsonify(stats)


def get_current_example() -> Optional[Dict[str, Any]]:
    """Get currently selected example."""
    if not current_examples or current_example_index >= len(current_examples):
        return None
    
    example = current_examples[current_example_index]
    example_id = example.get('id', f'example_{current_example_index}')
    
    # Add decision info if available
    if example_id in manual_decisions:
        example['manual_decision'] = manual_decisions[example_id]
    
    return example


def convert_example_for_injection(example: Dict[str, Any]) -> Dict[str, Any]:
    """Convert LateBench example to error injection format."""
    
    solution = example.get('solution', {})
    steps = solution.get('steps', [])
    
    # Create solution text
    solution_text = '\n'.join([
        step.get('content', str(step)) if isinstance(step, dict) else str(step)
        for step in steps
    ])
    
    return {
        'problem': example.get('problem', {}).get('statement', ''),
        'solution': solution_text,
        'answer': solution.get('final_answer', 'See solution'),
        'id': example.get('id', 'unknown')
    }


def save_manual_decisions():
    """Save manual decisions to persistent storage."""
    try:
        decisions_file = Path('./data/manual_decisions.json')
        decisions_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(decisions_file, 'w') as f:
            json.dump(manual_decisions, f, indent=2)
            
    except Exception as e:
        print(f"Warning: Could not save manual decisions: {e}")


def load_manual_decisions():
    """Load manual decisions from persistent storage."""
    global manual_decisions
    
    try:
        decisions_file = Path('./data/manual_decisions.json')
        if decisions_file.exists():
            with open(decisions_file, 'r') as f:
                manual_decisions = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load manual decisions: {e}")
        manual_decisions = {}


if __name__ == '__main__':
    # Load existing decisions
    load_manual_decisions()
    
    print("🚀 LateBench Simplified Dashboard")
    print("📊 Features: Dataset loading, Error injection, Manual review, Decision tracking")
    print("🌐 Open http://localhost:8000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=8000)