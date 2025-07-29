"""
Unified critic system for LateBench mathematical reasoning evaluation.
Core functionality: evaluate LateBenchExample objects for mathematical errors with step-level analysis.
"""

import os
import re
import time
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
from unified_schema import (
    LateBenchExample, LateBenchCriticPrediction, 
    create_timestamp
)

load_dotenv()




class MathCritic:
    """LLM-based critic for evaluating mathematical reasoning."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.rate_limit = int(os.getenv("REQUESTS_PER_MINUTE", 60))
        self.last_request = 0

    def evaluate_example(self, example: LateBenchExample, evaluation_mode: str = "auto") -> LateBenchExample:
        """Evaluate a LateBenchExample for mathematical errors and update critic predictions directly.
        
        Args:
            example: LateBenchExample to evaluate
            evaluation_mode: "original" (evaluate original solution), 
                           "injected" (evaluate error-injected solution),
                           "auto" (evaluate injected if available, otherwise original)
        
        Returns:
            Updated LateBenchExample with appropriate critic_predictions populated
        """
        
        start_time = time.time()
        
        # Determine which solution to evaluate and which field to update
        if evaluation_mode == "original":
            solution_to_evaluate = example.solution
            target_field = "original"
        elif evaluation_mode == "injected":
            if example.error_injection.has_errors and example.error_injection.injected_solution:
                solution_to_evaluate = example.error_injection.injected_solution
                target_field = "injected"
            else:
                raise ValueError("No error-injected solution available for evaluation")
        else:  # auto mode
            if example.error_injection.has_errors and example.error_injection.injected_solution:
                solution_to_evaluate = example.error_injection.injected_solution
                target_field = "injected"
            else:
                solution_to_evaluate = example.solution
                target_field = "original"
        
        # Clean and format inputs - NO access to correct answers
        clean_problem = self._clean_text(example.problem.statement)
        formatted_steps = self._format_steps([step.content for step in solution_to_evaluate.steps])
        
        # Create evaluation prompt (no correct answer leak)
        prompt = self._create_evaluation_prompt(clean_problem, formatted_steps)
        
        try:
            # Apply rate limiting
            self._apply_rate_limit()
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            raw_response = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            # Parse response
            parsed_result = self._parse_response(raw_response)
            
            # Create LateBenchCriticPrediction and update appropriate field
            prediction = LateBenchCriticPrediction(
                has_errors=parsed_result['has_errors'],
                error_steps=parsed_result['error_steps'],
                confidence_scores={},  # Can be enhanced later
                explanations=parsed_result['explanations'],
                processing_time=processing_time,
                model_version=self.model,
                prediction_timestamp=create_timestamp(),
                success=True
            )
            
            # Update the appropriate field
            if target_field == "original":
                example.critic_predictions_original = prediction
            else:
                example.critic_predictions_injected = prediction
            
            return example
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            # Create failed prediction
            failed_prediction = LateBenchCriticPrediction(
                has_errors=False,
                error_steps=[],
                confidence_scores={},
                explanations={0: f"Evaluation failed: {str(e)}"},
                processing_time=processing_time,
                model_version=self.model,
                prediction_timestamp=create_timestamp(),
                success=False
            )
            
            # Update the appropriate field
            if target_field == "original":
                example.critic_predictions_original = failed_prediction
            else:
                example.critic_predictions_injected = failed_prediction
            
            return example

    def evaluate_batch(self, examples: List[LateBenchExample], evaluation_mode: str = "auto", max_workers: int = 8) -> List[LateBenchExample]:
        """Evaluate multiple LateBenchExample objects in parallel and update them directly."""
        
        completed = 0
        
        print(f"Evaluating {len(examples)} examples with {max_workers} workers (mode: {evaluation_mode})...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._evaluate_single_thread_safe, example, evaluation_mode): i
                for i, example in enumerate(examples)
            }
            
            # Process completed jobs
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    updated_example = future.result()
                    examples[index] = updated_example  # Update in place
                    completed += 1
                    
                    if completed % 10 == 0:
                        print(f"Completed {completed}/{len(examples)} evaluations")
                        
                except Exception as e:
                    print(f"Error evaluating example {examples[index].id}: {e}")
                    # Create failed prediction for this example based on evaluation mode
                    failed_prediction = LateBenchCriticPrediction(
                        has_errors=False,
                        error_steps=[],
                        confidence_scores={},
                        explanations={0: f"Batch evaluation failed: {str(e)}"},
                        processing_time=0.0,
                        model_version=self.model,
                        prediction_timestamp=create_timestamp(),
                        success=False
                    )
                    
                    # Determine which field to update based on evaluation mode
                    if evaluation_mode == "original":
                        examples[index].critic_predictions_original = failed_prediction
                    elif evaluation_mode == "injected":
                        examples[index].critic_predictions_injected = failed_prediction
                    else:  # auto mode
                        if examples[index].error_injection.has_errors and examples[index].error_injection.injected_solution:
                            examples[index].critic_predictions_injected = failed_prediction
                        else:
                            examples[index].critic_predictions_original = failed_prediction
                    
                    completed += 1
        
        print(f"Batch evaluation complete: {len(examples)} examples updated")
        return examples

    def _evaluate_single_thread_safe(self, example: LateBenchExample, evaluation_mode: str = "auto") -> LateBenchExample:
        """Thread-safe single evaluation for parallel processing."""
        return self.evaluate_example(example, evaluation_mode)

    def _get_system_prompt(self) -> str:
        """Get system prompt for critic evaluation."""
        return """You are an expert mathematician evaluating mathematical solutions for errors.

Examine each step for:
- Logical reasoning flaws
- Incorrect theorem applications  
- Domain restriction violations
- Invalid assumptions or generalizations
- Calculation errors
- Missing or incorrect case analysis

Evaluate each step independently based on mathematical accuracy and logical validity."""

    def _create_evaluation_prompt(self, problem: str, steps: str) -> str:
        """Create evaluation prompt for the critic."""
        return f"""Evaluate this mathematical solution for errors.

**Problem:** {problem}

**Solution Steps:**
{steps}

**Instructions:**
- Examine each step for mathematical errors
- Focus on logical reasoning, calculations, and theorem applications
- Mark steps with invalid assumptions, domain violations, or flawed logic

**Output Format:**
- If all steps are correct: "Conclusion: no error"
- If errors found:
  Conclusion: yes
  Error Step Number: [step number]
  Explanation: [detailed error explanation]
  [repeat for each error]

**Your evaluation:**"""

    def _format_steps(self, step_contents: List[str]) -> str:
        """Format solution step contents for critic evaluation."""
        formatted_steps = []
        
        for i, content in enumerate(step_contents, 1):
            # Clean LaTeX formatting
            clean_content = self._clean_text(content)
            formatted_steps.append(f"Step {i}: {clean_content}")
        
        return '\n\n'.join(formatted_steps)

    def _clean_text(self, text: str) -> str:
        """Clean LaTeX escaping and formatting artifacts."""
        if not text:
            return ""
        
        # Fix common LaTeX escaping issues
        text = text.replace('\\\\(', '$').replace('\\\\)', '$')
        text = text.replace('\\\\[', '$$').replace('\\\\]', '$$')
        text = text.replace('\\textbackslash{}', '\\').replace('\\textbackslash', '\\')
        
        # Fix double-escaped LaTeX commands
        text = re.sub(r'\\\\([a-zA-Z]+)\{', r'\\\1{', text)
        text = re.sub(r'\\\\([a-zA-Z]+)\s', r'\\\1 ', text)
        
        return text.strip()

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse critic response into structured format."""
        
        result = {
            'has_errors': False,
            'error_steps': [],
            'explanations': {}
        }
        
        # Check for "no error" conclusion
        if "conclusion: no error" in response.lower():
            return result
        
        # Look for "yes" conclusion indicating errors
        if "conclusion: yes" not in response.lower():
            return result
        
        result['has_errors'] = True
        
        # Extract error steps and explanations
        lines = response.split('\n')
        current_step = None
        current_explanation = []
        
        for line in lines:
            line = line.strip()
            
            # Look for error step numbers
            step_match = re.search(r'error step number:\s*(\d+)', line, re.IGNORECASE)
            if step_match:
                # Save previous step if exists
                if current_step is not None and current_explanation:
                    result['explanations'][current_step] = ' '.join(current_explanation).strip()
                
                # Start new step
                current_step = int(step_match.group(1))
                result['error_steps'].append(current_step)
                current_explanation = []
                continue
            
            # Look for explanations
            explanation_match = re.search(r'explanation:\s*(.+)', line, re.IGNORECASE)
            if explanation_match and current_step is not None:
                current_explanation.append(explanation_match.group(1))
                continue
            
            # Continue multi-line explanation
            if current_step is not None and line and not line.startswith('Error Step Number:'):
                if line.lower().startswith('explanation:'):
                    current_explanation.append(line[12:].strip())
                elif current_explanation:
                    current_explanation.append(line)
        
        # Don't forget the last step
        if current_step is not None and current_explanation:
            result['explanations'][current_step] = ' '.join(current_explanation).strip()
        
        # Remove duplicates and sort
        result['error_steps'] = sorted(list(set(result['error_steps'])))
        
        return result

    def _apply_rate_limit(self):
        """Apply rate limiting between API calls."""
        if self.rate_limit <= 0:
            return
        
        elapsed = time.time() - self.last_request
        min_interval = 60.0 / self.rate_limit
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request = time.time()


# Convenience functions
def evaluate_single_example(example: LateBenchExample, model: str = "gpt-4o-mini", evaluation_mode: str = "auto") -> LateBenchExample:
    """Quick evaluation of a single LateBenchExample object - updates appropriate critic_predictions field directly."""
    
    critic = MathCritic(model=model)
    return critic.evaluate_example(example, evaluation_mode)