"""
Flask web dashboard for manual inspection of adversarial mathematical examples.
"""

import sys
import os
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from typing import Dict, Any, Optional

from dashboard.utils import DashboardData, analyze_critic_performance
from src.critic import LLMCritic, evaluate_single_example
from src.adapters.latebench_adapter import LateBenchAdapter, EvaluationPipeline

# Configure logging (will be properly set up by run_dashboard.py)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')
app.secret_key = 'latebench_dashboard_secret'

# Global data manager
dashboard_data = None
latebench_adapter = None


def initialize_data():
    """Initialize dashboard data."""
    global dashboard_data, latebench_adapter
    
    # Try different possible paths for the results file
    possible_paths = [
        # New MATH Level 5 natural errors dataset (highest priority)
        "./data/datasets/latebench_math_level5_natural_errors.json",
        # Core datasets
        "./data/datasets/latebench_prm800k_raw.json",
        "./data/datasets/latebench_numinamath_raw.json",
        # Fallback paths
        "./data/small_experiment_results.json",
        "../data/small_experiment_results.json",
        "data/small_experiment_results.json"
    ]
    
    results_file = None
    for path in possible_paths:
        if os.path.exists(path):
            results_file = path
            logger.info(f"Loading data from: {path}")
            break
    
    if results_file:
        dashboard_data = DashboardData(results_file)
        dashboard_data.load_critic_results()  # Load any existing critic results
    else:
        print("Warning: No results file found. Dashboard will start empty.")
        dashboard_data = DashboardData()
    
    # Initialize LateBench adapter for batch operations
    latebench_adapter = LateBenchAdapter(enable_dashboard_integration=True)


@app.route('/')
def index():
    """Main dashboard page."""
    if not dashboard_data or not dashboard_data.examples:
        return render_template('empty.html')
    
    # Get statistics
    stats = dashboard_data.get_statistics()
    
    # Get first example to display
    current_example = dashboard_data.examples[0] if dashboard_data.examples else None
    
    return render_template('index.html',
                         examples=dashboard_data.examples,
                         current_example=current_example,
                         stats=stats,
                         error_types=dashboard_data.get_error_types(),
                         available_datasets=dashboard_data.get_available_datasets(),
                         current_dataset=dashboard_data.get_current_dataset_info())


@app.route('/example/<example_id>')
def view_example(example_id):
    """View a specific example."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    # Add critic performance analysis if available
    if example.get('critic_result'):
        example['critic_analysis'] = analyze_critic_performance(example)
    
    stats = dashboard_data.get_statistics()
    
    return render_template('index.html',
                         examples=dashboard_data.examples,
                         current_example=example,
                         stats=stats,
                         error_types=dashboard_data.get_error_types(),
                         available_datasets=dashboard_data.get_available_datasets(),
                         current_dataset=dashboard_data.get_current_dataset_info())


@app.route('/api/examples')
def api_examples():
    """API endpoint to get all examples."""
    return jsonify({
        'examples': dashboard_data.examples,
        'stats': dashboard_data.get_statistics()
    })


@app.route('/api/example/<example_id>')
def api_example(example_id):
    """API endpoint to get a specific example."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    # Add critic analysis if available
    if example.get('critic_result'):
        example['critic_analysis'] = analyze_critic_performance(example)
    
    return jsonify(example)


@app.route('/api/run_critic/<example_id>', methods=['POST'])
def api_run_critic(example_id):
    """Run critic evaluation on a specific example."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    try:
        # Get the raw example data for critic
        raw_example = {
            'original_problem': {
                'problem': example['problem'],
                'parsed_steps': [step['content'] for step in example.get('original_steps', [])]
            },
            'modified_solution': {
                'steps': [
                    {
                        'step_num': step.get('step_number', step.get('number', i+1)),
                        'content': step['content'],
                        'modified': step.get('is_modified', False),
                        'error': step.get('is_error', False)
                    }
                    for i, step in enumerate(example.get('modified_solution', {}).get('steps', []))
                ]
            }
        }
        
        # Run critic evaluation
        critic_result = evaluate_single_example(raw_example)
        
        # Store result
        dashboard_data.add_critic_result(example_id, critic_result)
        dashboard_data.save_critic_results()
        
        # Return result with analysis
        result_dict = critic_result.to_dict()
        result_dict['analysis'] = analyze_critic_performance(
            dashboard_data.get_example(example_id)
        )
        
        return jsonify({
            'success': True,
            'critic_result': result_dict
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/filter')
def api_filter():
    """Filter examples based on criteria."""
    error_type = request.args.get('error_type')
    min_steps = request.args.get('min_steps', type=int)
    max_steps = request.args.get('max_steps', type=int)
    has_critic = request.args.get('has_critic')
    decision_filter = request.args.get('decision_filter')  # 'all', 'hide_no', 'yes_only', etc.
    
    # Start with decision-based filtering
    if decision_filter == 'hide_no':
        filtered_examples = dashboard_data.get_examples_by_decision(exclude_decision='no')
    elif decision_filter == 'yes_only':
        filtered_examples = dashboard_data.get_examples_by_decision(decision='yes')
    elif decision_filter == 'maybe_only':
        filtered_examples = dashboard_data.get_examples_by_decision(decision='maybe')
    elif decision_filter == 'no_only':
        filtered_examples = dashboard_data.get_examples_by_decision(decision='no')
    elif decision_filter == 'undecided_only':
        filtered_examples = dashboard_data.get_examples_by_decision(decision=None)
    else:
        filtered_examples = dashboard_data.examples
    
    # Apply filters
    if error_type and error_type != 'all':
        filtered_examples = [
            ex for ex in filtered_examples 
            if ex['error_info']['type'] == error_type
        ]
    
    if min_steps is not None:
        filtered_examples = [
            ex for ex in filtered_examples
            if ex['original_solution']['num_steps'] >= min_steps
        ]
    
    if max_steps is not None:
        filtered_examples = [
            ex for ex in filtered_examples
            if ex['original_solution']['num_steps'] <= max_steps
        ]
    
    if has_critic is not None:
        has_critic_bool = has_critic.lower() == 'true'
        filtered_examples = [
            ex for ex in filtered_examples
            if bool(ex.get('critic_result')) == has_critic_bool
        ]
    
    return jsonify({
        'examples': filtered_examples,
        'count': len(filtered_examples)
    })


@app.route('/api/stats')
def api_stats():
    """Get dashboard statistics."""
    return jsonify(dashboard_data.get_statistics())


@app.route('/api/save_suggestion/<example_id>', methods=['POST'])
def api_save_suggestion(example_id):
    """Save custom error suggestion for an example."""
    try:
        data = request.get_json()
        suggestion = data.get('suggestion', '').strip()
        
        if not suggestion:
            return jsonify({'success': False, 'error': 'Empty suggestion'}), 400
        
        dashboard_data.update_custom_suggestion(example_id, suggestion)
        dashboard_data.save_manual_injection_data()
        
        return jsonify({
            'success': True,
            'message': 'Suggestion saved successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/manual_injection/<example_id>', methods=['POST'])
def api_manual_injection(example_id):
    """Run manual error injection with custom suggestions."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    try:
        data = request.get_json()
        user_remarks = data.get('user_remarks', '').strip()
        custom_suggestions = dashboard_data.get_manual_injection_data(example_id)['custom_suggestions']
        
        if not custom_suggestions:
            return jsonify({
                'success': False,
                'error': 'No custom error suggestions available. Please add a suggestion first.'
            }), 400
        
        # Import error injector
        from src.error_injector import AdversarialErrorInjector
        
        # Create injector instance
        injector = AdversarialErrorInjector()
        
        # Prepare the original problem data for injection
        # Use modified_solution steps since that's where the dashboard stores the actual solution steps
        steps = example.get('modified_solution', {}).get('steps', [])
        if not steps:
            # Fallback to original_steps if available
            steps = example.get('original_steps', [])
        
        raw_example = {
            'problem': example['problem'],
            'solution': '\n'.join([step['content'] for step in steps]),
            'answer': example.get('modified_solution', {}).get('final_answer', 'No answer provided')
        }
        
        # Use the most recent custom suggestion as error type preference
        custom_suggestion = custom_suggestions[-1] if custom_suggestions else None
        
        # Run error injection with custom suggestion
        injection_result = injector.inject_error_with_custom_suggestion(
            raw_example,
            custom_suggestion=custom_suggestion,
            max_retries=3
        )
        
        # Store the attempt
        attempt_data = {
            'user_remarks': user_remarks,
            'injection_result': injection_result.__dict__ if injection_result else {},
            'custom_suggestion': custom_suggestion
        }
        
        dashboard_data.add_injection_attempt(example_id, attempt_data)
        dashboard_data.save_manual_injection_data()
        
        if injection_result and injection_result.success:
            return jsonify({
                'success': True,
                'injection_result': {
                    'modified_solution': injection_result.modified_solution,
                    'error_analysis': injection_result.error_analysis,
                    'error_explanation': injection_result.error_explanation,
                    'attempt_number': len(dashboard_data.get_manual_injection_data(example_id)['injection_attempts'])
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': injection_result.error_message if injection_result else 'Injection failed'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/set_decision/<example_id>', methods=['POST'])
def api_set_decision(example_id):
    """Set final decision (yes/maybe/no) for an example."""
    try:
        print(f"DEBUG: Received decision request for example_id: {example_id}")
        data = request.get_json()
        print(f"DEBUG: Request data: {data}")
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data received'
            }), 400
        
        decision = data.get('decision', '').strip().lower()
        print(f"DEBUG: Decision: {decision}")
        
        if decision not in ['yes', 'maybe', 'no']:
            return jsonify({
                'success': False,
                'error': 'Invalid decision. Must be yes, maybe, or no.'
            }), 400
        
        # Check if example exists
        example = dashboard_data.get_example(example_id)
        if not example:
            return jsonify({
                'success': False,
                'error': f'Example {example_id} not found'
            }), 404
        
        dashboard_data.set_final_decision(example_id, decision)
        dashboard_data.save_manual_injection_data()
        
        print(f"DEBUG: Successfully set decision {decision} for {example_id}")
        
        return jsonify({
            'success': True,
            'decision': decision,
            'message': f'Decision set to "{decision}"'
        })
    
    except Exception as e:
        print(f"DEBUG: Error in set_decision: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/manual_data/<example_id>')
def api_manual_data(example_id):
    """Get manual injection data for an example."""
    try:
        manual_data = dashboard_data.get_manual_injection_data(example_id)
        return jsonify({
            'success': True,
            'data': manual_data
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/injection_history/<example_id>')
def api_injection_history(example_id):
    """Get manual injection history for an example."""
    try:
        manual_data = dashboard_data.get_manual_injection_data(example_id)
        return jsonify({
            'success': True,
            'history': manual_data.get('injection_attempts', []),
            'suggestions': manual_data.get('custom_suggestions', []),
            'final_decision': manual_data.get('final_decision'),
            'decision_timestamp': manual_data.get('decision_timestamp')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/datasets')
def api_datasets():
    """Get available datasets and current dataset info."""
    try:
        available_datasets = dashboard_data.get_available_datasets()
        current_info = dashboard_data.get_current_dataset_info()
        
        return jsonify({
            'success': True,
            'available_datasets': available_datasets,
            'current_dataset': current_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/switch_dataset', methods=['POST'])
def api_switch_dataset():
    """Switch to a different dataset and problem type."""
    try:
        data = request.get_json()
        dataset_name = data.get('dataset_name')
        problem_type = data.get('problem_type', 'all')
        
        if not dataset_name:
            return jsonify({
                'success': False,
                'error': 'Dataset name is required'
            }), 400
        
        success = dashboard_data.switch_dataset(dataset_name, problem_type)
        
        if success:
            return jsonify({
                'success': True,
                'current_dataset': dashboard_data.get_current_dataset_info(),
                'stats': dashboard_data.get_statistics(),
                'examples_count': len(dashboard_data.examples)
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to dataset {dataset_name} ({problem_type})'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Batch Evaluation API Endpoints

@app.route('/api/batch_evaluate', methods=['POST'])
def api_batch_evaluate():
    """Start batch evaluation of current dataset."""
    try:
        data = request.get_json() or {}
        
        # Get current dataset info
        current_dataset = dashboard_data.get_current_dataset_info()
        dataset_name = current_dataset['name']
        
        if dataset_name == 'None':
            return jsonify({
                'success': False,
                'error': 'No dataset currently loaded'
            }), 400
        
        # Configuration from request
        config = {
            'model_version': data.get('model_version', 'gpt-4o-mini'),
            'max_concurrent': data.get('max_concurrent', 10),
            'use_caching': data.get('use_caching', True),
            'compute_deltabench_metrics': data.get('compute_deltabench_metrics', True)
        }
        
        # Create evaluation pipeline
        pipeline = EvaluationPipeline(
            dataset_name=dataset_name,
            model_version=config['model_version'],
            max_concurrent=config['max_concurrent'],
            use_caching=config['use_caching'],
            compute_deltabench_metrics=config['compute_deltabench_metrics'],
            save_results=True
        )
        
        # Start batch evaluation in background thread
        def run_batch_evaluation():
            try:
                summary = latebench_adapter.evaluate_dataset(pipeline)
                # Store evaluation ID for progress tracking
                app.config['last_batch_summary'] = summary
                logger.info(f"Batch evaluation completed: {summary.batch_id}")
            except Exception as e:
                logger.error(f"Batch evaluation failed: {e}")
                app.config['batch_evaluation_error'] = str(e)
        
        import threading
        eval_thread = threading.Thread(target=run_batch_evaluation)
        eval_thread.daemon = True
        eval_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Batch evaluation started',
            'dataset_name': dataset_name,
            'config': config
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/batch_status')
def api_batch_status():
    """Get status of batch evaluation."""
    try:
        # Check if evaluation completed
        if 'last_batch_summary' in app.config:
            summary = app.config['last_batch_summary']
            return jsonify({
                'status': 'completed',
                'batch_id': summary.batch_id,
                'total_examples': summary.total_examples,
                'successful_evaluations': summary.successful_evaluations,
                'failed_evaluations': summary.failed_evaluations,
                'evaluation_end': summary.evaluation_end,
                'deltabench_metrics': summary.deltabench_metrics.to_dict() if summary.deltabench_metrics else None
            })
        
        # Check for errors
        if 'batch_evaluation_error' in app.config:
            error = app.config['batch_evaluation_error']
            del app.config['batch_evaluation_error']  # Clear error
            return jsonify({
                'status': 'error',
                'error': error
            })
        
        # Still running or not started
        return jsonify({
            'status': 'running',
            'message': 'Batch evaluation in progress'
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500


@app.route('/api/evaluation_summary/<dataset_name>')
def api_evaluation_summary(dataset_name):
    """Get comprehensive evaluation summary for a dataset."""
    try:
        summary = latebench_adapter.get_evaluation_summary(dataset_name)
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/batch_history')
def api_batch_history():
    """Get history of batch evaluations."""
    try:
        # Get current dataset name
        current_dataset = dashboard_data.get_current_dataset_info()
        dataset_name = current_dataset['name']
        
        if dataset_name == 'None':
            return jsonify([])
        
        # Get batch summaries from adapter
        summaries = latebench_adapter.critic_store.get_batch_summaries(dataset_name, limit=20)
        
        # Convert to JSON-serializable format
        history = []
        for summary in summaries:
            history.append({
                'batch_id': summary.batch_id,
                'dataset_name': summary.dataset_name,
                'model_version': summary.model_version,
                'total_examples': summary.total_examples,
                'successful_evaluations': summary.successful_evaluations,
                'failed_evaluations': summary.failed_evaluations,
                'evaluation_start': summary.evaluation_start,
                'evaluation_end': summary.evaluation_end,
                'deltabench_metrics': summary.deltabench_metrics.to_dict() if summary.deltabench_metrics else None
            })
        
        return jsonify(history)
        
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/refresh_critic_results', methods=['POST'])
def api_refresh_critic_results():
    """Refresh critic results from storage for current dataset."""
    try:
        # Get current dataset info
        current_dataset = dashboard_data.get_current_dataset_info()
        dataset_name = current_dataset['name']
        
        if dataset_name == 'None':
            return jsonify({
                'success': False,
                'error': 'No dataset currently loaded'
            }), 400
        
        # Get stored results
        stored_results = latebench_adapter.get_existing_results(dataset_name)
        
        # Update dashboard data with stored results
        updated_count = 0
        for example_id, stored_result in stored_results.items():
            if stored_result.critic_result:
                dashboard_data.add_critic_result(example_id, stored_result.critic_result)
                updated_count += 1
        
        return jsonify({
            'success': True,
            'updated_count': updated_count,
            'total_stored_results': len(stored_results)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/export/<example_id>')
def export_example(example_id):
    """Export example as JSON."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    # Create export data
    export_data = {
        'example_id': example_id,
        'title': example['title'],
        'problem': example['problem'],
        'original_solution': example['original_solution'],
        'modified_solution': example['modified_solution'],
        'error_info': example['error_info'],
        'critic_result': example.get('critic_result'),
        'critic_analysis': analyze_critic_performance(example) if example.get('critic_result') else None
    }
    
    return jsonify(export_data)


@app.route('/next')
def next_example():
    """Navigate to the next example."""
    current_id = request.args.get('current', type=int, default=0)
    
    if not dashboard_data.examples:
        return redirect(url_for('index'))
    
    # Find current example index
    current_index = 0
    for i, ex in enumerate(dashboard_data.examples):
        if ex.get('id') == current_id:
            current_index = i
            break
    
    # Get next example (or loop back to first)
    next_index = (current_index + 1) % len(dashboard_data.examples)
    next_id = dashboard_data.examples[next_index]['id']
    
    return redirect(url_for('view_example', example_id=next_id))


@app.route('/prev')
def prev_example():
    """Navigate to the previous example."""
    current_id = request.args.get('current', type=int, default=0)
    
    if not dashboard_data.examples:
        return redirect(url_for('index'))
    
    # Find current example index
    current_index = 0
    for i, ex in enumerate(dashboard_data.examples):
        if ex.get('id') == current_id:
            current_index = i
            break
    
    # Get previous example (or loop back to last)
    prev_index = (current_index - 1) % len(dashboard_data.examples)
    prev_id = dashboard_data.examples[prev_index]['id']
    
    return redirect(url_for('view_example', example_id=prev_id))


def create_templates():
    """Create template files if they don't exist."""
    
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    
    # Create directories
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)
    
    # Create base template
    base_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LateBench Dashboard</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
            displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
        }
    };
    </script>
</head>
<body>
    <div id="app">
        {% block content %}{% endblock %}
    </div>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>'''
    
    with open(os.path.join(templates_dir, 'base.html'), 'w') as f:
        f.write(base_template)
    
    print("Created template files")


if __name__ == '__main__':
    # Initialize data
    initialize_data()
    
    # Create template files
    create_templates()
    
    # Check if we have data
    if not dashboard_data or not dashboard_data.examples:
        print("Warning: No examples loaded. Make sure to run experiments first.")
        print("Expected file: ./data/small_experiment_results.json")
    else:
        print(f"Dashboard loaded with {len(dashboard_data.examples)} examples")
    
    # Run Flask app
    print("Starting LateBench Dashboard...")
    print("Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=8000)