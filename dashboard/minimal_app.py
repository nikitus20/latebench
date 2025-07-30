#!/usr/bin/env python3
"""
Minimal LateBench Dashboard
Simple dashboard that loads a single LateBench dataset and provides basic functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from flask import Flask, render_template, request, jsonify
import json
from typing import List, Dict, Any, Optional

# Import our core components
from core.data_loader import LateBenchDataLoader
from core.error_injector import ErrorInjector
from core.critic import MathCritic, evaluate_single_example
from data_processing.unified_schema import LateBenchExample, LateBenchManualDecision
from utils.storage import save_examples_to_file, load_examples_from_file

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'latebench_minimal_dashboard'

# Global state
examples: List[LateBenchExample] = []
current_index = 0
PERSISTENT_FILE = "dashboard/data/examples_with_decisions.json"

def load_dataset():
    """Load the LateBench dataset, preferring persistent file if it exists."""
    global examples
    
    try:
        # First try to load from persistent file
        if os.path.exists(PERSISTENT_FILE):
            print(f"📂 Loading examples from persistent file: {PERSISTENT_FILE}")
            examples = load_examples_from_file(PERSISTENT_FILE)
            print(f"✅ Loaded {len(examples)} examples with persistent decisions")
            return True
        
        # If no persistent file, load from original dataset
        print("📂 No persistent file found, loading fresh from dataset...")
        loader = LateBenchDataLoader()
        examples = loader.load_dataset("prm800k", "complete")
        
        if not examples:
            print(f"❌ No examples loaded from prm800k dataset")
            return False
            
        print(f"✅ Loaded {len(examples)} LateBenchExample objects from PRM800K dataset")
        
        # Save to persistent file for future loads
        save_examples_to_persistent_file()
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

def save_examples_to_persistent_file():
    """Save current examples to persistent file."""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(PERSISTENT_FILE), exist_ok=True)
        save_examples_to_file(examples, PERSISTENT_FILE)
        print(f"💾 Saved {len(examples)} examples to persistent file")
    except Exception as e:
        print(f"❌ Error saving to persistent file: {e}")

@app.route('/')
def index():
    """Main dashboard page."""
    if not examples:
        return render_template('minimal_dashboard.html', 
                             error="No dataset loaded", 
                             examples=[], 
                             current_example=None,
                             current_index=0,
                             total_examples=0)
    
    current_example = examples[current_index] if current_index < len(examples) else None
    
    # Get current decision if it exists
    current_decision = None
    if current_example and current_example.manual_decision and current_example.manual_decision.decision:
        current_decision = current_example.manual_decision.decision
    
    return render_template('minimal_dashboard.html',
                         examples=[],  # Don't pass all examples to template
                         current_example=current_example,
                         current_index=current_index,
                         total_examples=len(examples),
                         current_decision=current_decision)

@app.route('/api/navigate/<direction>')
def navigate(direction):
    """Navigate between examples."""
    global current_index
    
    if direction == 'next':
        current_index = min(current_index + 1, len(examples) - 1)
    elif direction == 'prev':
        current_index = max(current_index - 1, 0)
    elif direction == 'first':
        current_index = 0
    elif direction == 'last':
        current_index = len(examples) - 1
    
    current_example = examples[current_index] if current_index < len(examples) else None
    
    return jsonify({
        'success': True,
        'current_index': current_index,
        'current_example': current_example,
        'total_examples': len(examples)
    })

@app.route('/api/jump_to/<int:target_index>')
def jump_to(target_index):
    """Jump to a specific example by index."""
    global current_index
    
    if target_index < 0 or target_index >= len(examples):
        return jsonify({'success': False, 'error': 'Invalid example index'}), 400
    
    current_index = target_index
    current_example = examples[current_index]
    
    return jsonify({
        'success': True,
        'current_index': current_index,
        'current_example': current_example,
        'total_examples': len(examples)
    })

@app.route('/api/inject_error', methods=['POST'])
def inject_error():
    """Inject error into current example."""
    if not examples or current_index >= len(examples):
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    custom_suggestion = data.get('custom_suggestion')
    target_error_step = data.get('target_error_step')
    
    current_example = examples[current_index]
    
    try:
        # Current example is already a LateBenchExample object - no conversion needed
        injector = ErrorInjector()
        
        # Set manual instructions if provided
        if custom_suggestion or target_error_step:
            injector.set_manual_instruction(
                current_example, 
                manual_suggestion=custom_suggestion,
                target_error_step=target_error_step
            )
        
        result_example = injector.inject_error(current_example)
        
        # Update the example in our examples list
        examples[current_index] = result_example
        
        # Save to persistent file
        save_examples_to_persistent_file()
        
        if result_example.error_injection.success:
            return jsonify({
                'success': True,
                'injection_result': {
                    'injected_solution': result_example.error_injection.injected_solution.to_dict() if result_example.error_injection.injected_solution else None,
                    'error_info': result_example.error_injection.error_info,
                    'base_prompt': result_example.error_injection.base_prompt,
                    'manual_suggestion': result_example.error_injection.manual_suggestion
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': result_example.error_injection.error_info.get('error', 'Unknown error')
            })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/evaluate_with_critic', methods=['POST'])
def evaluate_with_critic():
    """Evaluate current example with critic."""
    if not examples or current_index >= len(examples):
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    evaluation_mode = data.get('evaluation_mode', 'auto')
    
    current_example = examples[current_index]
    
    try:
        # Current example is already a LateBenchExample object - no conversion needed
        updated_example = evaluate_single_example(current_example, evaluation_mode=evaluation_mode)
        
        # Update the example in our examples list
        examples[current_index] = updated_example
        
        # Save to persistent file
        save_examples_to_persistent_file()
        
        # Return the appropriate critic prediction based on evaluation mode
        if evaluation_mode == "original":
            critic_result = updated_example.critic_predictions_original.to_dict() if updated_example.critic_predictions_original else None
        elif evaluation_mode == "injected":
            critic_result = updated_example.critic_predictions_injected.to_dict() if updated_example.critic_predictions_injected else None
        else:  # auto mode - return whichever was evaluated
            if updated_example.critic_predictions_injected:
                critic_result = updated_example.critic_predictions_injected.to_dict()
            elif updated_example.critic_predictions_original:
                critic_result = updated_example.critic_predictions_original.to_dict()
            else:
                critic_result = None
        
        return jsonify({
            'success': True,
            'critic_result': critic_result
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_decision', methods=['POST'])
def save_decision():
    """Save manual decision for current example."""
    if not examples or current_index >= len(examples):
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    decision = data.get('decision')  # 'yes', 'maybe', 'no'
    notes = data.get('notes', '')
    
    if decision not in ['yes', 'maybe', 'no']:
        return jsonify({'success': False, 'error': 'Invalid decision'}), 400
    
    try:
        current_example = examples[current_index]
        
        # Create or update the manual decision 
        from datetime import datetime
        current_example.manual_decision = LateBenchManualDecision(
            decision=decision,
            notes=notes.strip() if notes.strip() else None,
            annotator="dashboard_user",  # Could be made configurable later
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
        # Update the example in our examples list
        examples[current_index] = current_example
        
        # Save to persistent file
        save_examples_to_persistent_file()
        
        return jsonify({
            'success': True,
            'decision': decision,
            'notes': notes,
            'example_id': current_example.id
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_suggestion', methods=['POST'])
def save_suggestion():
    """Save custom suggestion and target step without injecting error."""
    if not examples or current_index >= len(examples):
        return jsonify({'success': False, 'error': 'No example selected'}), 400
    
    data = request.get_json()
    custom_suggestion = data.get('custom_suggestion', '').strip()
    target_error_step = data.get('target_error_step')
    
    current_example = examples[current_index]
    
    try:
        # Update the suggestion and target step
        current_example.error_injection.manual_suggestion = custom_suggestion if custom_suggestion else None
        current_example.error_injection.target_error_step = target_error_step if target_error_step else None
        
        # Update the example in our examples list
        examples[current_index] = current_example
        
        # Save to persistent file
        save_examples_to_persistent_file()
        
        return jsonify({
            'success': True,
            'message': 'Suggestion saved successfully',
            'suggestion': custom_suggestion,
            'target_step': target_error_step
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    """Get current session statistics."""
    # Count decisions from example objects
    decisions_with_manual = [ex for ex in examples if ex.manual_decision and ex.manual_decision.decision]
    decisions_made = len(decisions_with_manual)
    
    # Count by decision type
    yes_count = sum(1 for ex in examples if ex.manual_decision and ex.manual_decision.decision == 'yes')
    maybe_count = sum(1 for ex in examples if ex.manual_decision and ex.manual_decision.decision == 'maybe')
    no_count = sum(1 for ex in examples if ex.manual_decision and ex.manual_decision.decision == 'no')
    
    stats = {
        'total_examples': len(examples),
        'current_index': current_index,
        'decisions_made': decisions_made,
        'decisions_by_type': {
            'yes': yes_count,
            'maybe': maybe_count,
            'no': no_count
        }
    }
    
    return jsonify(stats)

def create_minimal_template():
    """Create minimal template for the dashboard."""
    template_dir = Path(__file__).parent / "templates"
    template_dir.mkdir(exist_ok=True)
    
    template_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LateBench Minimal Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        .header { text-align: center; margin-bottom: 20px; padding: 20px; background: #2c3e50; color: white; border-radius: 8px; }
        .content { display: grid; grid-template-columns: 300px 1fr; gap: 20px; }
        .sidebar { background: #f8f9fa; padding: 20px; border-radius: 8px; }
        .main-content { background: #fff; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
        .section { margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .btn { background: #3498db; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin: 2px; }
        .btn:hover { background: #2980b9; }
        .btn-success { background: #27ae60; }
        .btn-warning { background: #f39c12; }
        .btn-danger { background: #e74c3c; }
        .problem-display { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px 0; white-space: pre-wrap; }
        .solution-step { background: white; border: 1px solid #ddd; padding: 10px; margin: 5px 0; border-radius: 4px; }
        .solution-step.error { background: #ffeaa7; border-color: #fdcb6e; }
        textarea, input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin: 5px 0; box-sizing: border-box; }
        .error-message { background: #e74c3c; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
        .success-message { background: #27ae60; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
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
    <div class="container">
        <div class="header">
            <h1>🧮 LateBench Minimal Dashboard</h1>
            <p>NuminaMath Dataset - {{ total_examples }} Examples</p>
        </div>
        
        {% if error %}
            <div class="error-message">{{ error }}</div>
        {% else %}
            <div class="content">
                <div class="sidebar">
                    <div class="section">
                        <h3>Navigation</h3>
                        <div>Example {{ current_index + 1 }} of {{ total_examples }}</div>
                        <div style="margin: 10px 0;">
                            <button class="btn" onclick="navigate('first')">⏮️ First</button>
                            <button class="btn" onclick="navigate('prev')">⬅️ Prev</button>
                            <button class="btn" onclick="navigate('next')">➡️ Next</button>
                            <button class="btn" onclick="navigate('last')">⏭️ Last</button>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>Error Injection</h3>
                        <textarea id="customSuggestion" placeholder="Custom error suggestion (optional)..." rows="3"></textarea>
                        <button class="btn" onclick="injectError()">🎲 Inject Error</button>
                    </div>
                    
                    <div class="section">
                        <h3>Critic Evaluation</h3>
                        <button class="btn" onclick="evaluateWithCritic()">🔍 Run Critic</button>
                    </div>
                    
                    <div class="section">
                        <h3>Manual Decision</h3>
                        <div>
                            <button class="btn btn-success" onclick="saveDecision('yes')">✅ Yes</button>
                            <button class="btn btn-warning" onclick="saveDecision('maybe')">⚠️ Maybe</button>
                            <button class="btn btn-danger" onclick="saveDecision('no')">❌ No</button>
                        </div>
                        <textarea id="decisionNotes" placeholder="Notes..." rows="2"></textarea>
                    </div>
                    
                    <div class="section">
                        <h3>Stats</h3>
                        <div id="stats">Decisions: {{ manual_decisions|length }}</div>
                    </div>
                </div>
                
                <div class="main-content">
                    {% if current_example %}
                        <div class="section">
                            <h3>Problem Statement</h3>
                            <div class="problem-display">{{ current_example.problem.statement }}</div>
                        </div>
                        
                        <div class="section">
                            <h3>Original Solution ({{ current_example.solution.steps|length }} steps)</h3>
                            <div id="originalSolution">
                                {% for step in current_example.solution.steps %}
                                    <div class="solution-step">
                                        <strong>Step {{ loop.index }}:</strong> {{ step.content }}
                                    </div>
                                {% endfor %}
                            </div>
                        </div>
                        
                        <div class="section" id="injectedSolution" style="display: none;">
                            <h3>Modified Solution (After Error Injection)</h3>
                            <div id="modifiedSolutionDisplay"></div>
                        </div>
                        
                        <div class="section" id="criticResult" style="display: none;">
                            <h3>Critic Evaluation</h3>
                            <div id="criticResultDisplay"></div>
                        </div>
                    {% else %}
                        <div class="section">
                            <h3>No Example Available</h3>
                            <p>No examples to display.</p>
                        </div>
                    {% endif %}
                </div>
            </div>
        {% endif %}
    </div>

    <script>
        function navigate(direction) {
            fetch(`/api/navigate/${direction}`)
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    location.reload();
                }
            })
            .catch(error => showError('Navigation error: ' + error));
        }

        function injectError() {
            const customSuggestion = document.getElementById('customSuggestion').value;
            
            showLoading('Injecting error...');
            
            fetch('/api/inject_error', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({custom_suggestion: customSuggestion})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    displayInjectionResult(data.injection_result);
                    showSuccess('Error injection successful!');
                } else {
                    showError('Error injection failed: ' + data.error);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Error injection error: ' + error);
            });
        }

        function evaluateWithCritic() {
            showLoading('Running critic evaluation...');
            
            fetch('/api/evaluate_with_critic', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({})
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    displayCriticResult(data.critic_result);
                    showSuccess('Critic evaluation complete!');
                } else {
                    showError('Critic evaluation failed: ' + data.error);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Critic evaluation error: ' + error);
            });
        }

        function saveDecision(decision) {
            const notes = document.getElementById('decisionNotes').value;
            
            fetch('/api/save_decision', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({decision: decision, notes: notes})
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    showSuccess(`Decision saved: ${decision}`);
                    updateStats();
                } else {
                    showError('Failed to save decision: ' + data.error);
                }
            })
            .catch(error => showError('Decision save error: ' + error));
        }

        function displayInjectionResult(result) {
            const section = document.getElementById('injectedSolution');
            const display = document.getElementById('modifiedSolutionDisplay');
            
            if (result.injected_solution && result.injected_solution.steps) {
                let html = '';
                result.injected_solution.steps.forEach(step => {
                    const cssClass = step.is_error ? 'solution-step error' : 'solution-step';
                    html += `<div class="${cssClass}">
                        <strong>Step ${step.step_number}:</strong> ${step.content}
                        ${step.is_error ? ' <strong>[ERROR]</strong>' : ''}
                    </div>`;
                });
                display.innerHTML = html;
                section.style.display = 'block';
                
                if (window.MathJax) {
                    MathJax.typesetPromise([display]);
                }
            }
        }

        function displayCriticResult(result) {
            const section = document.getElementById('criticResult');
            const display = document.getElementById('criticResultDisplay');
            
            let html = `
                <div><strong>Has Errors:</strong> ${result.has_errors ? 'Yes' : 'No'}</div>
                <div><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</div>
            `;
            
            if (result.explanations && Object.keys(result.explanations).length > 0) {
                html += '<div><strong>Explanations:</strong></div>';
                for (const [step, explanation] of Object.entries(result.explanations)) {
                    html += `<div class="solution-step"><strong>Step ${step}:</strong> ${explanation}</div>`;
                }
            }
            
            display.innerHTML = html;
            section.style.display = 'block';
        }

        function updateStats() {
            fetch('/api/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('stats').innerHTML = `
                    Decisions: ${data.decisions_made}/${data.total_examples}<br>
                    Yes: ${data.decisions_by_type.yes} Maybe: ${data.decisions_by_type.maybe} No: ${data.decisions_by_type.no}
                `;
            });
        }

        function showLoading(message) {
            document.body.style.cursor = 'wait';
        }

        function hideLoading() {
            document.body.style.cursor = 'default';
        }

        function showError(message) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = message;
            document.body.appendChild(errorDiv);
            setTimeout(() => errorDiv.remove(), 5000);
        }

        function showSuccess(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'success-message';
            successDiv.textContent = message;
            document.body.appendChild(successDiv);
            setTimeout(() => successDiv.remove(), 3000);
        }

        // Update stats on page load
        document.addEventListener('DOMContentLoaded', updateStats);
    </script>
</body>
</html>'''

    with open(template_dir / "minimal_dashboard.html", 'w') as f:
        f.write(template_content)
    
    print("✅ Created minimal dashboard template")

if __name__ == '__main__':
    print("🚀 Starting LateBench Minimal Dashboard")
    
    # Load dataset
    if not load_dataset():
        print("❌ Failed to load dataset")
        exit(1)
    
    # Create template - disabled to use enhanced template
    # create_minimal_template()
    
    print(f"📊 Loaded {len(examples)} examples")
    print("🌐 Starting server on http://localhost:8080")
    
    app.run(debug=True, host='0.0.0.0', port=8080)