# NuminaMath Dataset Adversarial Error Injection Project

## Project Overview

This project aims to create a dataset of mathematical problems with intentionally injected late-appearing logical errors to test the capabilities of process reward models and reasoning critics. We'll use the NuminaMath dataset as our source of high-quality mathematical problems and solutions.

## Phase 1: Dataset Acquisition and Analysis

### 1.1 Download NuminaMath Dataset

```python
# Installation
pip install datasets huggingface_hub

# Download script
from datasets import load_dataset

# Load the NuminaMath dataset
dataset = load_dataset("AI-MO/NuminaMath-CoT")

# Save locally for offline work
dataset.save_to_disk("./numinamath_local")
```

### 1.2 Dataset Structure Analysis

```python
# Analyze dataset structure
def analyze_dataset_structure(dataset):
    """
    Expected structure based on NuminaMath:
    - problem: Mathematical problem statement
    - solution: Step-by-step solution in natural language
    - answer: Final numerical/algebraic answer
    - level: Difficulty level (1-5)
    - type: Problem type/category
    """
    
    # Get basic statistics
    print(f"Total examples: {len(dataset['train'])}")
    print(f"Features: {dataset['train'].features}")
    
    # Analyze solution lengths
    solution_lengths = []
    for example in dataset['train']:
        steps = example['solution'].split('\n')
        solution_lengths.append(len(steps))
    
    print(f"Average solution length: {np.mean(solution_lengths):.2f} steps")
    print(f"Max solution length: {max(solution_lengths)} steps")
    
    return solution_lengths
```

### 1.3 Filter for Long Solutions

```python
def filter_long_solutions(dataset, min_steps=8):
    """
    Filter dataset to only include problems with sufficiently long solutions
    for meaningful late error injection
    """
    filtered_examples = []
    
    for example in dataset:
        steps = parse_solution_steps(example['solution'])
        if len(steps) >= min_steps:
            filtered_examples.append({
                **example,
                'parsed_steps': steps,
                'num_steps': len(steps)
            })
    
    return filtered_examples
```

## Phase 2: Error Injection Algorithm

### 2.1 Improved System Prompt for GPT-4

```
You are an expert mathematician tasked with creating adversarial examples for testing reasoning error detection models. Your goal is to introduce subtle logical errors late in mathematical solutions that would be challenging to detect.

TASK: Given a mathematical problem and its correct solution, introduce a logical reasoning error in the last 25% of the solution steps that:
1. Appears plausible at first glance
2. Uses valid mathematical notation and operations
3. Leads to an incorrect final answer through flawed reasoning (not just arithmetic mistakes)
4. Requires careful analysis to detect

IMPORTANT CONSTRAINTS:
- The error should be a LOGICAL REASONING ERROR, not a simple calculation mistake
- The error should maintain superficial coherence with the problem context
- All subsequent steps must be rewritten to follow from the erroneous step
- The final answer must be different from the correct answer

ERROR TYPES TO FOCUS ON:
1. Invalid generalizations or assumptions
2. Incorrect application of theorems or properties
3. Flawed case analysis or missing cases
4. Circular reasoning or assuming what needs to be proven
5. Incorrect logical implications or equivalences
```

### 2.2 User Prompt Template

```
Given the following mathematical problem and its correct solution, introduce a logical reasoning error in one of the last 25% of steps:

PROBLEM: {problem_statement}

CORRECT SOLUTION:
{numbered_steps}

CORRECT ANSWER: {correct_answer}

Please provide your response in the following JSON format:
{
    "error_injection_analysis": {
        "total_steps": <number>,
        "target_step_range": "<step numbers in last 25%>",
        "selected_error_step": <step number>,
        "error_type": "<specific type of logical error>",
        "error_rationale": "<why this step is suitable for error injection>"
    },
    "modified_solution": {
        "steps": [
            {"step_num": 1, "content": "<unchanged step 1>", "modified": false},
            {"step_num": 2, "content": "<unchanged step 2>", "modified": false},
            ...
            {"step_num": N, "content": "<modified step with error>", "modified": true, "error": true},
            {"step_num": N+1, "content": "<rewritten step following from error>", "modified": true},
            ...
        ],
        "final_answer": "<incorrect answer resulting from error>"
    },
    "error_explanation": {
        "what_changed": "<specific description of the logical flaw introduced>",
        "why_incorrect": "<mathematical explanation of why this reasoning is invalid>",
        "detection_difficulty": "<why this error might be hard to detect>"
    }
}
```

### 2.3 Implementation Code

```python
import openai
import json
from typing import List, Dict, Tuple

class AdversarialErrorInjector:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        
    def parse_solution_steps(self, solution: str) -> List[str]:
        """Parse solution into numbered steps"""
        # Handle various step formats
        steps = []
        lines = solution.strip().split('\n')
        current_step = []
        
        for line in lines:
            if line.strip() and (line[0].isdigit() or line.startswith('Step')):
                if current_step:
                    steps.append(' '.join(current_step))
                current_step = [line]
            elif line.strip():
                current_step.append(line)
        
        if current_step:
            steps.append(' '.join(current_step))
            
        return steps
    
    def inject_error(self, problem: Dict) -> Dict:
        """Inject a logical error into the solution"""
        steps = self.parse_solution_steps(problem['solution'])
        num_steps = len(steps)
        
        # Calculate last 25% range
        last_quarter_start = int(num_steps * 0.75)
        
        # Format numbered steps
        numbered_steps = '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
        
        # Create prompt
        user_prompt = self.user_prompt_template.format(
            problem_statement=problem['problem'],
            numbered_steps=numbered_steps,
            correct_answer=problem['answer']
        )
        
        # Call GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Add metadata
        result['original_problem'] = problem
        result['injection_metadata'] = {
            'num_original_steps': num_steps,
            'last_quarter_range': f"{last_quarter_start}-{num_steps}",
            'model_used': 'gpt-4-turbo-preview'
        }
        
        return result
```

### 2.4 Example Generation and Visualization

```python
def create_example_visualization(original_problem: Dict, injected_result: Dict) -> str:
    """Create a clear visualization of the error injection"""
    
    viz = f"""
# Problem: {original_problem['problem']}

## Original Correct Solution:
{original_problem['solution']}
**Correct Answer:** {original_problem['answer']}

---

## Modified Solution with Injected Error:

### Error Analysis:
- **Error Step:** Step {injected_result['error_injection_analysis']['selected_error_step']}
- **Error Type:** {injected_result['error_injection_analysis']['error_type']}
- **Why Selected:** {injected_result['error_injection_analysis']['error_rationale']}

### Modified Steps:
"""
    
    for step in injected_result['modified_solution']['steps']:
        if step['modified']:
            if step.get('error'):
                viz += f"\n**→ Step {step['step_num']} [ERROR INJECTED]:**\n{step['content']}\n"
            else:
                viz += f"\n**→ Step {step['step_num']} [MODIFIED]:**\n{step['content']}\n"
        else:
            viz += f"\nStep {step['step_num']}: {step['content']}\n"
    
    viz += f"""
**Incorrect Final Answer:** {injected_result['modified_solution']['final_answer']}

### Error Explanation:
- **What Changed:** {injected_result['error_explanation']['what_changed']}
- **Why It's Wrong:** {injected_result['error_explanation']['why_incorrect']}
- **Detection Difficulty:** {injected_result['error_explanation']['detection_difficulty']}
"""
    
    return viz
```

## Phase 3: Error Classification System

### 3.1 Logical Error Taxonomy for Mathematics

```python
LOGICAL_ERROR_TYPES = {
    "invalid_generalization": {
        "description": "Applying a specific case result to general case without justification",
        "example": "Because it works for n=2, it works for all n"
    },
    "theorem_misapplication": {
        "description": "Using a theorem outside its valid conditions",
        "example": "Applying Pythagorean theorem to non-right triangles"
    },
    "incomplete_case_analysis": {
        "description": "Missing important cases in proof by cases",
        "example": "Only considering positive values when negative are possible"
    },
    "circular_reasoning": {
        "description": "Using the conclusion to prove itself",
        "example": "Assuming what needs to be proven in the proof"
    },
    "false_equivalence": {
        "description": "Treating non-equivalent statements as equivalent",
        "example": "If A then B treated as equivalent to If B then A"
    },
    "domain_restriction_violation": {
        "description": "Performing operations outside valid domain",
        "example": "Dividing by expression that could be zero"
    },
    "quantifier_confusion": {
        "description": "Mixing up 'for all' and 'there exists'",
        "example": "Proving existence when universality is needed"
    },
    "invalid_substitution": {
        "description": "Substituting under conditions where it's not valid",
        "example": "Substituting into inequality without considering sign"
    }
}
```

### 3.2 Error Detection Evaluation Framework

```python
class ErrorDetectionEvaluator:
    def __init__(self):
        self.error_types = LOGICAL_ERROR_TYPES
        
    def evaluate_critic_model(self, model, test_examples):
        """Evaluate a critic model's ability to detect injected errors"""
        results = {
            'total': len(test_examples),
            'correct_detections': 0,
            'false_positives': 0,
            'missed_errors': 0,
            'by_error_type': {et: {'detected': 0, 'total': 0} for et in self.error_types},
            'by_position': {'early': {'detected': 0, 'total': 0}, 
                           'middle': {'detected': 0, 'total': 0},
                           'late': {'detected': 0, 'total': 0}}
        }
        
        for example in test_examples:
            prediction = model.predict(example)
            actual_error_step = example['error_injection_analysis']['selected_error_step']
            error_type = example['error_injection_analysis']['error_type']
            
            # Update statistics
            # ... (implementation details)
            
        return results
```

## Phase 4: Dataset Creation Pipeline

### 4.1 Full Pipeline Implementation

```python
def create_adversarial_dataset(
    source_dataset,
    num_examples=1000,
    min_steps=8,
    error_injection_rate=1.0
):
    """Create dataset with adversarial examples"""
    
    # Initialize components
    injector = AdversarialErrorInjector(api_key=OPENAI_API_KEY)
    
    # Filter suitable problems
    long_problems = filter_long_solutions(source_dataset, min_steps)
    
    # Sample problems
    sampled_problems = random.sample(long_problems, 
                                   min(num_examples, len(long_problems)))
    
    # Create adversarial examples
    adversarial_examples = []
    for problem in tqdm(sampled_problems):
        try:
            injected = injector.inject_error(problem)
            adversarial_examples.append(injected)
            
            # Save intermediate results
            if len(adversarial_examples) % 10 == 0:
                save_checkpoint(adversarial_examples)
                
        except Exception as e:
            print(f"Error processing problem: {e}")
            continue
    
    return adversarial_examples
```

### 4.2 Quality Control and Validation

```python
def validate_injected_errors(examples):
    """Ensure injected errors meet quality criteria"""
    validation_results = []
    
    for ex in examples:
        checks = {
            'error_in_last_quarter': check_error_position(ex),
            'different_final_answer': check_answer_difference(ex),
            'logical_error_type': check_error_is_logical(ex),
            'steps_follow_error': check_consistency_after_error(ex),
            'maintains_plausibility': check_surface_plausibility(ex)
        }
        
        validation_results.append({
            'example_id': ex.get('id'),
            'all_checks_passed': all(checks.values()),
            'failed_checks': [k for k, v in checks.items() if not v],
            'checks': checks
        })
    
    return validation_results
```

## Next Steps

1. **Run initial experiments** with 100 examples to refine prompts
2. **Analyze error distribution** and adjust injection strategy
3. **Create evaluation benchmark** with human annotations
4. **Test existing PRMs** on the adversarial dataset
5. **Iterate on error types** based on detection difficulty

## Expected Deliverables

1. **Adversarial NuminaMath dataset** with 1000+ examples
2. **Error injection codebase** with configurable parameters
3. **Evaluation framework** for testing critic models
4. **Analysis report** on error types and detection difficulty
5. **Baseline results** from existing PRMs and critic models
