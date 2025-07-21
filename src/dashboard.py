"""
Flask web dashboard for manual inspection of adversarial mathematical examples.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
from typing import Dict, Any, Optional

from dashboard_utils import DashboardData, analyze_critic_performance
from critic import LLMCritic, evaluate_single_example

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'latebench_dashboard_secret'

# Global data manager
dashboard_data = None


def initialize_data():
    """Initialize dashboard data."""
    global dashboard_data
    
    # Try different possible paths for the results file
    possible_paths = [
        "./data/small_experiment_results.json",
        "../data/small_experiment_results.json",
        "data/small_experiment_results.json"
    ]
    
    results_file = None
    for path in possible_paths:
        if os.path.exists(path):
            results_file = path
            break
    
    if results_file:
        dashboard_data = DashboardData(results_file)
        dashboard_data.load_critic_results()  # Load any existing critic results
    else:
        print("Warning: No results file found. Dashboard will start empty.")
        dashboard_data = DashboardData()


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
                         error_types=dashboard_data.get_error_types())


@app.route('/example/<int:example_id>')
def view_example(example_id: int):
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
                         error_types=dashboard_data.get_error_types())


@app.route('/api/examples')
def api_examples():
    """API endpoint to get all examples."""
    return jsonify({
        'examples': dashboard_data.examples,
        'stats': dashboard_data.get_statistics()
    })


@app.route('/api/example/<int:example_id>')
def api_example(example_id: int):
    """API endpoint to get a specific example."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    # Add critic analysis if available
    if example.get('critic_result'):
        example['critic_analysis'] = analyze_critic_performance(example)
    
    return jsonify(example)


@app.route('/api/run_critic/<int:example_id>', methods=['POST'])
def api_run_critic(example_id: int):
    """Run critic evaluation on a specific example."""
    example = dashboard_data.get_example(example_id)
    
    if not example:
        return jsonify({'error': 'Example not found'}), 404
    
    try:
        # Get the raw example data for critic
        raw_example = {
            'original_problem': {
                'problem': example['problem'],
                'parsed_steps': [step['content'] for step in example['original_solution']['steps']]
            },
            'modified_solution': {
                'steps': [
                    {
                        'step_num': step['number'],
                        'content': step['content'],
                        'modified': step['is_modified'],
                        'error': step['is_error']
                    }
                    for step in example['modified_solution']['steps']
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


@app.route('/export/<int:example_id>')
def export_example(example_id: int):
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
    
    app.run(debug=True, host='0.0.0.0', port=5000)