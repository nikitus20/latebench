"""
Unified error injection system for LateBench mathematical reasoning.
Core functionality: inject natural mathematical errors using unified LateBench format.
"""

import os
import json
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import openai
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add data_processing to path for unified schema imports
sys.path.insert(0, str(Path(__file__).parent.parent / "data_processing"))
try:
    from unified_schema import (
        LateBenchExample, LateBenchSolution, LateBenchStep, 
        create_timestamp
    )
except ImportError:
    # Fallback for IDE path resolution
    from src.data_processing.unified_schema import (
        LateBenchExample, LateBenchSolution, LateBenchStep, 
        create_timestamp
    )

load_dotenv()


class ErrorInjector:
    """Inject subtle mathematical errors using unified LateBench format."""
    
    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.rate_limit = int(os.getenv("REQUESTS_PER_MINUTE", 60))
        self.last_request = 0
        
        # Base prompt is now handled by unified prompt method

    def set_manual_instruction(self, example: LateBenchExample, manual_suggestion: Optional[str] = None, target_error_step: Optional[int] = None) -> None:
        """Set manual instruction parameters in the LateBench example."""
        if manual_suggestion:
            example.error_injection.manual_suggestion = manual_suggestion
        if target_error_step:
            example.error_injection.target_error_step = target_error_step

    def inject_error(self, example: LateBenchExample) -> LateBenchExample:
        """Inject a mathematical error into LateBench example - reads configuration from example data."""
        
        # Create unified prompt (handles both manual and automatic modes based on example data)
        prompt = self._create_unified_prompt(example)
        
        # Call API with rate limiting
        try:
            self._apply_rate_limit()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            result_data = json.loads(response.choices[0].message.content)
            
            # Get the error step number from the analysis
            error_step_num = result_data['error_injection_analysis']['selected_error_step']
            
            # Create complete solution: original steps up to error + modified continuation
            complete_steps = []
            
            # Add original steps before the error (unchanged)
            for i in range(error_step_num - 1):
                original_step = example.solution.steps[i]
                complete_steps.append(LateBenchStep(
                    step_number=i + 1,
                    content=original_step.content,
                    importance=original_step.importance,
                    reasoning_type=original_step.reasoning_type,
                    is_modified=False,
                    is_error=False
                ))
            
            # Add the modified steps (error + continuation) from API response
            for i, step_data in enumerate(result_data['modified_solution']['steps']):
                step_number = error_step_num + i
                complete_steps.append(LateBenchStep(
                    step_number=step_number,
                    content=step_data['content'],
                    importance=step_data.get('importance', 'medium'),
                    reasoning_type=step_data.get('reasoning_type', 'unknown'),
                    is_modified=step_data.get('modified', False),
                    is_error=step_data.get('error', False)
                ))
            
            injected_solution = LateBenchSolution(
                steps=complete_steps,
                final_answer=result_data['modified_solution']['final_answer'],
                total_steps=len(complete_steps),
                solution_method=example.solution.solution_method
            )
            
            # Extract injected error step numbers from the solution
            injected_error_steps = [step.step_number for step in complete_steps if step.is_error]
            
            # Update error injection data
            example.error_injection.has_errors = True
            example.error_injection.injected_solution = injected_solution
            example.error_injection.base_prompt = "unified_prompt_v1"
            example.error_injection.injection_timestamp = create_timestamp()
            example.error_injection.success = True
            example.error_injection.error_info = {
                'error_analysis': result_data.get('error_injection_analysis', {}),
                'error_explanation': result_data.get('error_explanation', {}),
                'model_used': self.model,
                'step_targeting': 'manual' if example.error_injection.target_error_step else 'automatic'
            }
            
            # Update injected error steps field (preserve original_error_steps)
            example.injected_error_steps = injected_error_steps
            
            return example
            
        except Exception as e:
            # Mark as failed injection but return the example
            example.error_injection.success = False
            example.error_injection.error_info = {"error": str(e)}
            example.error_injection.injection_timestamp = create_timestamp()
            # Keep injected_error_steps empty for failed injections
            example.injected_error_steps = []
            return example

    def inject_batch(self, examples: List[LateBenchExample], max_workers: int = 4) -> List[LateBenchExample]:
        """Inject errors into multiple LateBench examples in parallel.
        
        Args:
            examples: List of LateBenchExample objects to process
            max_workers: Number of parallel workers (lower than critic due to API rate limits)
        
        Returns:
            List of updated LateBenchExample objects with error injections
        """
        
        completed = 0
        failed = 0
        
        print(f"Injecting errors into {len(examples)} examples with {max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_index = {
                executor.submit(self._inject_error_thread_safe, example): i
                for i, example in enumerate(examples)
            }
            
            # Process completed jobs
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                
                try:
                    updated_example = future.result()
                    examples[index] = updated_example  # Update in place
                    completed += 1
                    
                    if updated_example.error_injection.success:
                        if completed % 5 == 0:
                            print(f"Completed {completed}/{len(examples)} error injections")
                    else:
                        failed += 1
                        print(f"Failed injection for example {updated_example.id}: {updated_example.error_injection.error_info.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    failed += 1
                    print(f"Critical error processing example {examples[index].id}: {e}")
                    # Mark as failed injection
                    examples[index].error_injection.success = False
                    examples[index].error_injection.error_info = {"error": f"Batch processing failed: {str(e)}"}
                    examples[index].error_injection.injection_timestamp = create_timestamp()
                    completed += 1
        
        successful = completed - failed
        print(f"Batch error injection complete: {successful}/{len(examples)} successful, {failed} failed")
        return examples

    def _inject_error_thread_safe(self, example: LateBenchExample) -> LateBenchExample:
        """Thread-safe single error injection for parallel processing."""
        return self.inject_error(example)

    def _create_unified_prompt(self, example: LateBenchExample) -> str:
        """Create unified prompt for error injection - handles both manual and automatic modes."""
        
        total_steps = len(example.solution.steps)
        
        # Determine target step and create clear instructions
        if example.error_injection.target_error_step is not None:
            target_step = example.error_injection.target_error_step
            if target_step < 1 or target_step > total_steps:
                raise ValueError(f"Target error step {target_step} is out of range (1-{total_steps})")
            
            step_instruction = f"EXACTLY at step {target_step} (manually specified)"
            range_explanation = f"You must modify step {target_step} to contain the error."
        else:
            # Automatic mode - final third (33%)
            late_start = max(1, int(total_steps * 0.67))
            step_instruction = f"in the final third of the solution (steps {late_start}-{total_steps})"
            range_explanation = f"You must choose ONE step between {late_start} and {total_steps} to modify with the error."
        
        # Format solution steps with clear numbering
        steps_text = self._format_steps_for_prompt([step.content for step in example.solution.steps])
        
        # Build unified prompt with clear continuation requirements
        prompt = f"""You are creating mathematical solutions with subtle errors for educational testing.

CRITICAL REQUIREMENTS:
- Modified solution must appear COMPLETELY NORMAL and CONFIDENT
- No uncertainty, hesitation, or qualifying language  
- NO ERROR MARKERS like "[ERROR]" or "[WRONG]" - the error must be invisible
- Present each step as mathematically sound and obvious
- Use definitive statements: "Therefore", "Thus", "We have", "This gives us"
- Never use: "seems", "appears", "might", "could be", "possibly"
- Write the error step exactly as if it were correct mathematics

PROBLEM: {example.problem.statement}

CORRECT SOLUTION ({total_steps} steps):
{steps_text}

CORRECT ANSWER: {example.solution.final_answer}

ERROR INJECTION TASK:
1. INJECT ERROR: {range_explanation}
2. CONTINUE SOLUTION: After the error step, continue with ALL remaining steps that would logically follow from the erroneous step
3. FINAL ANSWER: Provide the incorrect final answer that results from the error

ERROR PLACEMENT: The error must appear {step_instruction}

SOLUTION CONTINUATION REQUIREMENT:
- If you modify step N, you MUST provide steps N+1 through {total_steps}
- All subsequent steps must follow naturally from the modified (erroneous) step
- Maintain the same mathematical style and step format as the original
- Each subsequent step should be a logical consequence of the error
- The solution must reach a definitive (but incorrect) final answer

ERROR TYPES TO USE (must be subtle and convincing):
- Logical errors: Skip critical cases in analysis, assume conditions that don't hold
- Domain violations: Apply theorems/formulas outside their valid scope
- Invalid inferences: Draw conclusions that don't follow from premises
- Flawed algebraic manipulations: Incorrect factoring, distribution, or cancellation
- Geometric misconceptions: Misapply properties of shapes, distances, or angles
- Sign errors: Incorrect handling of positive/negative values
- Boundary condition errors: Miss edge cases or boundary behaviors

The error should look like a reasonable mathematical step that could fool an expert on first glance."""

        # Add manual guidance if provided
        if example.error_injection.manual_suggestion:
            prompt += f"""

EXTREMELY IMPORTANT GUIDANCE - Read, understand, and implement this specific instruction:
{example.error_injection.manual_suggestion}

This guidance takes priority over general instructions above. Use it to determine the exact error type, approach, or focus for the injection."""

        # Add JSON format specification
        prompt += f"""

RESPONSE: Valid JSON only, no other text.

JSON FORMAT:
{{
    "error_injection_analysis": {{
        "total_steps": {total_steps},
        "selected_error_step": <step_number_where_error_was_injected>,
        "error_type": "<error_category>",
        "error_rationale": "<educational_value>"
    }},
    "modified_solution": {{
        "steps": [
            {{"content": "<modified_step_that_contains_the_subtle_error_but_appears_correct>", "importance": "high", "reasoning_type": "logical", "modified": true, "error": true}},
            {{"content": "<next_step_logically_following_from_the_error>", "importance": "medium", "reasoning_type": "calculation", "modified": false, "error": false}},
            {{"content": "<another_continuation_step>", "importance": "medium", "reasoning_type": "logical", "modified": false, "error": false}},
            ...continue until the final step
        ],
        "final_answer": "<incorrect_final_answer>"
    }},
    "error_explanation": {{
        "what_changed": "<description_of_what_was_modified_in_the_error_step>",
        "why_incorrect": "<mathematical_explanation_of_why_the_error_leads_to_wrong_answer>", 
        "detection_hints": "<clues_that_would_help_someone_identify_this_error>"
    }}
}}

IMPORTANT: Only return steps starting from the error step through the end. If you inject error at step N:
- First step in your response should be the modified step N (with "modified": true, "error": true)
- Continue with steps N+1, N+2, ... through step {total_steps} (all following from the error)
- Do NOT repeat steps 1 through N-1 - we already have those from the original solution
- NEVER include "[ERROR]", "[WRONG]", or any error markers in the step content - it must look completely normal"""
        
        return prompt

    def _format_steps_for_prompt(self, steps: List[str]) -> str:
        """Format solution steps for prompt display."""
        return '\n'.join([f"Step {i+1}: {step}" for i, step in enumerate(steps)])

    def _apply_rate_limit(self):
        """Apply rate limiting between API calls."""
        if self.rate_limit <= 0:
            return
        
        elapsed = time.time() - self.last_request
        min_interval = 60.0 / self.rate_limit
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request = time.time()