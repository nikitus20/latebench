"""
Adversarial error injection system for mathematical reasoning.
"""

import os
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import openai
from dotenv import load_dotenv

from error_types import MATH_ERROR_TAXONOMY, LogicalErrorCategory

# Load environment variables
load_dotenv()


@dataclass
class InjectionResult:
    """Result of error injection operation."""
    success: bool
    original_problem: Dict[str, Any]
    modified_solution: Dict[str, Any]
    error_analysis: Dict[str, Any]
    error_explanation: Dict[str, Any]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None


class AdversarialErrorInjector:
    """Inject logical errors into mathematical solutions using GPT-4."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        self.error_taxonomy = MATH_ERROR_TAXONOMY
        
        # Rate limiting
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", 60))
        self.last_request_time = 0
        
        # Prompt templates
        self.system_prompt = self._create_system_prompt()
        self.user_prompt_template = self._create_user_prompt_template()
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for error injection."""
        return """You are an expert mathematician tasked with creating adversarial examples for testing reasoning error detection models. Your goal is to introduce subtle logical errors late in mathematical solutions that would be challenging to detect.

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
6. Domain restriction violations
7. Quantifier confusion
8. Invalid substitutions

RESPONSE FORMAT: You must respond with valid JSON only. No other text before or after the JSON."""

    def _create_user_prompt_template(self) -> str:
        """Create user prompt template."""
        return """Given the following mathematical problem and its correct solution, introduce a logical reasoning error in one of the last 25% of steps:

PROBLEM: {problem_statement}

CORRECT SOLUTION:
{numbered_steps}

CORRECT ANSWER: {correct_answer}

Please provide your response in the following JSON format:
{{
    "error_injection_analysis": {{
        "total_steps": <number>,
        "target_step_range": "<step numbers in last 25%>",
        "selected_error_step": <step number>,
        "error_type": "<specific type of logical error>",
        "error_rationale": "<why this step is suitable for error injection>"
    }},
    "modified_solution": {{
        "steps": [
            {{"step_num": 1, "content": "<unchanged step 1>", "modified": false}},
            {{"step_num": 2, "content": "<unchanged step 2>", "modified": false}},
            ...
            {{"step_num": N, "content": "<modified step with error>", "modified": true, "error": true}},
            {{"step_num": N+1, "content": "<rewritten step following from error>", "modified": true}},
            ...
        ],
        "final_answer": "<incorrect answer resulting from error>"
    }},
    "error_explanation": {{
        "what_changed": "<specific description of the logical flaw introduced>",
        "why_incorrect": "<mathematical explanation of why this reasoning is invalid>", 
        "detection_difficulty": "<why this error might be hard to detect>"
    }}
}}"""

    def parse_solution_steps(self, solution: str) -> List[str]:
        """Parse solution into numbered steps."""
        if not solution or not isinstance(solution, str):
            return []
        
        # Handle various step formats
        steps = []
        lines = solution.strip().split('\n')
        current_step = []
        
        step_indicators = ['step', 'Step', 'STEP', '1.', '2.', '3.', '4.', '5.']
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new step
            is_new_step = False
            if any(line.startswith(indicator) for indicator in step_indicators):
                is_new_step = True
            elif line[0].isdigit() and ('.' in line[:5] or ')' in line[:5]):
                is_new_step = True
            
            if is_new_step and current_step:
                steps.append(' '.join(current_step))
                current_step = [line]
            elif is_new_step:
                current_step = [line]
            else:
                current_step.append(line)
        
        # Don't forget the last step
        if current_step:
            steps.append(' '.join(current_step))
        
        # If no clear step structure found, split by sentences/paragraphs
        if len(steps) <= 1:
            # Split by double newlines (paragraphs)
            paragraphs = solution.split('\n\n')
            if len(paragraphs) > 1:
                steps = [p.strip() for p in paragraphs if p.strip()]
            else:
                # Split by sentences
                import re
                sentences = re.split(r'[.!?]+', solution)
                steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
        
        return [step for step in steps if step.strip()]
    
    def _rate_limit(self):
        """Implement rate limiting."""
        if self.requests_per_minute <= 0:
            return
        
        time_since_last = time.time() - self.last_request_time
        min_interval = 60.0 / self.requests_per_minute
        
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def inject_error(self, problem: Dict[str, Any], 
                    error_type_preference: Optional[str] = None,
                    max_retries: int = 3) -> InjectionResult:
        """Inject a logical error into the solution."""
        
        # Parse solution steps
        if 'solution' not in problem:
            return InjectionResult(
                success=False,
                original_problem=problem,
                modified_solution={},
                error_analysis={},
                error_explanation={},
                metadata={},
                error_message="No solution found in problem"
            )
        
        steps = self.parse_solution_steps(problem['solution'])
        num_steps = len(steps)
        
        if num_steps < 4:
            return InjectionResult(
                success=False,
                original_problem=problem,
                modified_solution={},
                error_analysis={},
                error_explanation={},
                metadata={},
                error_message=f"Solution too short ({num_steps} steps), need at least 4"
            )
        
        # Calculate last 25% range
        last_quarter_start = max(1, int(num_steps * 0.75))
        
        # Format numbered steps
        numbered_steps = '\\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])
        
        # Create enhanced prompt with error type hint if provided
        enhanced_system_prompt = self.system_prompt
        if error_type_preference:
            error_info = self.error_taxonomy.get_prompt_examples(error_type_preference)
            if error_info:
                enhanced_system_prompt += f"""

SPECIFIC ERROR TYPE TO INJECT: {error_info['error_type']}
- Description: {error_info['description']}  
- Example context: {error_info['example']}
- Focus on: {error_info['context_hint']}"""
        
        # Create user prompt
        user_prompt = self.user_prompt_template.format(
            problem_statement=problem.get('problem', 'No problem statement'),
            numbered_steps=numbered_steps,
            correct_answer=problem.get('answer', 'No answer provided')
        )
        
        # Attempt injection with retries
        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()
                
                # Call GPT-4
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": enhanced_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"}
                )
                
                # Parse response
                result_json = json.loads(response.choices[0].message.content)
                
                # Validate response structure
                required_keys = ['error_injection_analysis', 'modified_solution', 'error_explanation']
                if not all(key in result_json for key in required_keys):
                    raise ValueError(f"Missing required keys in response: {required_keys}")
                
                # Create successful result
                return InjectionResult(
                    success=True,
                    original_problem=problem,
                    modified_solution=result_json['modified_solution'],
                    error_analysis=result_json['error_injection_analysis'],
                    error_explanation=result_json['error_explanation'],
                    metadata={
                        'num_original_steps': num_steps,
                        'last_quarter_range': f"{last_quarter_start}-{num_steps}",
                        'model_used': self.model,
                        'attempt': attempt + 1,
                        'error_type_preference': error_type_preference
                    }
                )
                
            except json.JSONDecodeError as e:
                error_msg = f"JSON decode error on attempt {attempt + 1}: {e}"
                if attempt == max_retries - 1:
                    return InjectionResult(
                        success=False,
                        original_problem=problem,
                        modified_solution={},
                        error_analysis={},
                        error_explanation={},
                        metadata={},
                        error_message=error_msg
                    )
                    
            except Exception as e:
                error_msg = f"Error on attempt {attempt + 1}: {e}"
                if attempt == max_retries - 1:
                    return InjectionResult(
                        success=False,
                        original_problem=problem,
                        modified_solution={},
                        error_analysis={},
                        error_explanation={},
                        metadata={},
                        error_message=error_msg
                    )
        
        return InjectionResult(
            success=False,
            original_problem=problem,
            modified_solution={},
            error_analysis={},
            error_explanation={},
            metadata={},
            error_message="Max retries exceeded"
        )
    
    def batch_inject_errors(self, problems: List[Dict[str, Any]], 
                          error_distribution: Optional[Dict[str, float]] = None,
                          save_checkpoints: bool = True,
                          checkpoint_interval: int = 10) -> List[InjectionResult]:
        """Inject errors into a batch of problems."""
        
        results = []
        error_types = list(self.error_taxonomy.get_all_error_names())
        
        # Default error distribution (uniform)
        if error_distribution is None:
            error_distribution = {error_type: 1.0/len(error_types) for error_type in error_types}
        
        print(f"Starting batch injection for {len(problems)} problems...")
        
        for i, problem in enumerate(problems):
            print(f"Processing problem {i+1}/{len(problems)}...")
            
            # Select error type based on distribution
            error_type = random.choices(
                list(error_distribution.keys()),
                weights=list(error_distribution.values())
            )[0]
            
            # Inject error
            result = self.inject_error(problem, error_type_preference=error_type)
            results.append(result)
            
            if result.success:
                print(f"✓ Successfully injected {error_type} error")
            else:
                print(f"✗ Failed to inject error: {result.error_message}")
            
            # Save checkpoint
            if save_checkpoints and (i + 1) % checkpoint_interval == 0:
                checkpoint_path = f"./data/checkpoint_batch_{i+1}.json"
                self.save_results(results, checkpoint_path)
                print(f"Saved checkpoint at {checkpoint_path}")
        
        print(f"Batch complete. Success rate: {sum(1 for r in results if r.success)}/{len(results)}")
        return results
    
    def save_results(self, results: List[InjectionResult], filepath: str):
        """Save injection results to file."""
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append({
                'success': result.success,
                'original_problem': result.original_problem,
                'modified_solution': result.modified_solution,
                'error_analysis': result.error_analysis,
                'error_explanation': result.error_explanation,
                'metadata': result.metadata,
                'error_message': result.error_message
            })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Saved {len(results)} results to {filepath}")
    
    def load_results(self, filepath: str) -> List[InjectionResult]:
        """Load injection results from file."""
        
        if not os.path.exists(filepath):
            print(f"File {filepath} not found")
            return []
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = []
        for item in data:
            results.append(InjectionResult(
                success=item['success'],
                original_problem=item['original_problem'],
                modified_solution=item['modified_solution'],
                error_analysis=item['error_analysis'],
                error_explanation=item['error_explanation'],
                metadata=item['metadata'],
                error_message=item.get('error_message')
            ))
        
        print(f"Loaded {len(results)} results from {filepath}")
        return results