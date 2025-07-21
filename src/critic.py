"""
LLM-based critic for evaluating mathematical reasoning steps.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class CriticResult:
    """Result of critic evaluation."""
    has_errors: bool
    error_steps: List[int]
    explanations: Dict[int, str]
    raw_response: str
    model_used: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'has_errors': self.has_errors,
            'error_steps': self.error_steps,
            'explanations': self.explanations,
            'raw_response': self.raw_response,
            'model_used': self.model_used,
            'processing_time': self.processing_time
        }


class StepFormatter:
    """Utilities for cleaning and formatting mathematical solution steps."""
    
    @staticmethod
    def clean_latex_escaping(text: str) -> str:
        """Remove LaTeX escaping artifacts from text."""
        if not text:
            return ""
        
        # Remove common escaping patterns
        text = text.replace('\\\\(', '$(')
        text = text.replace('\\\\)', '$)')
        text = text.replace('\\\\[', '$[')
        text = text.replace('\\\\]', '$]')
        text = text.replace('\\textbackslash{}', '\\')
        text = text.replace('\\\\', '\\')
        
        # Clean up double backslashes in LaTeX commands
        text = re.sub(r'\\\\([a-zA-Z]+)', r'\\\1', text)
        
        # Fix common LaTeX commands
        text = text.replace('\\frac', '\\frac')
        text = text.replace('\\sqrt', '\\sqrt')
        text = text.replace('\\cos', '\\cos')
        text = text.replace('\\sin', '\\sin')
        
        return text.strip()
    
    @staticmethod
    def extract_step_content(step_dict: Dict[str, Any]) -> str:
        """Extract clean content from step dictionary."""
        if isinstance(step_dict, dict):
            content = step_dict.get('content', '')
        else:
            content = str(step_dict)
        
        return StepFormatter.clean_latex_escaping(content)
    
    @staticmethod
    def format_steps_for_critic(steps: List[Any]) -> str:
        """Format steps for critic evaluation."""
        formatted_steps = []
        
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                content = StepFormatter.extract_step_content(step)
            else:
                content = StepFormatter.clean_latex_escaping(str(step))
            
            formatted_steps.append(f"Step {i}: {content}")
        
        return '\n\n'.join(formatted_steps)
    
    @staticmethod
    def format_problem_statement(problem: str) -> str:
        """Clean and format problem statement."""
        return StepFormatter.clean_latex_escaping(problem)


class LLMCritic:
    """LLM-based critic for evaluating mathematical reasoning."""
    
    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = model
        
        # Rate limiting
        self.requests_per_minute = int(os.getenv("REQUESTS_PER_MINUTE", 60))
        self.last_request_time = 0
        
        # Prompt template adapted for mathematical steps
        self.system_prompt = """You are an expert mathematician tasked with evaluating mathematical solutions for errors. You will be given a mathematical problem and a solution broken into numbered steps.

Your task is to carefully examine each step for mathematical errors, including:
- Logical reasoning flaws
- Incorrect theorem applications  
- Domain restriction violations
- Invalid assumptions or generalizations
- Calculation errors
- Circular reasoning
- Missing or incorrect case analysis

Evaluate each step independently based solely on mathematical accuracy and logical validity."""

        self.user_prompt_template = """I will provide you with a mathematical problem and a solution divided into numbered steps. Your task is to evaluate each step for mathematical errors.

**Evaluation Criteria:**
- Evaluate each step independently for mathematical accuracy
- Focus on logical reasoning, calculation errors, theorem misapplication
- Mark steps with invalid assumptions, domain violations, or flawed logic
- If a step contains an error caused by a previous step's mistake, still mark it as erroneous

**Output Format:**
- If all steps are correct: "Conclusion: no error"
- If any step has errors:
  Conclusion: yes
  Error Step Number: [step number]
  Explanation: [detailed explanation of the mathematical error]
  [repeat for each erroneous step]

**Input:**
Problem: {problem}

Solution Steps:
{steps}

**Your evaluation:**"""

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
    
    def evaluate_solution(self, problem: str, solution_steps: List[Any], 
                         max_retries: int = 3) -> CriticResult:
        """Evaluate a mathematical solution for errors."""
        
        start_time = time.time()
        
        # Format inputs for critic
        clean_problem = StepFormatter.format_problem_statement(problem)
        formatted_steps = StepFormatter.format_steps_for_critic(solution_steps)
        
        # Create prompt
        user_prompt = self.user_prompt_template.format(
            problem=clean_problem,
            steps=formatted_steps
        )
        
        # Attempt evaluation with retries
        for attempt in range(max_retries):
            try:
                # Rate limiting
                self._rate_limit()
                
                # Call LLM
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent evaluation
                    max_tokens=2000
                )
                
                raw_response = response.choices[0].message.content
                processing_time = time.time() - start_time
                
                # Parse response
                critic_result = self._parse_critic_response(raw_response)
                
                return CriticResult(
                    has_errors=critic_result['has_errors'],
                    error_steps=critic_result['error_steps'],
                    explanations=critic_result['explanations'],
                    raw_response=raw_response,
                    model_used=self.model,
                    processing_time=processing_time
                )
                
            except Exception as e:
                if attempt == max_retries - 1:
                    # Final attempt failed
                    processing_time = time.time() - start_time
                    return CriticResult(
                        has_errors=False,
                        error_steps=[],
                        explanations={},
                        raw_response=f"Error: {str(e)}",
                        model_used=self.model,
                        processing_time=processing_time
                    )
                
                # Wait before retry
                time.sleep(2 ** attempt)
        
        # Should never reach here, but just in case
        processing_time = time.time() - start_time
        return CriticResult(
            has_errors=False,
            error_steps=[],
            explanations={},
            raw_response="Error: Max retries exceeded",
            model_used=self.model,
            processing_time=processing_time
        )
    
    def _parse_critic_response(self, response: str) -> Dict[str, Any]:
        """Parse critic response into structured format."""
        
        # Initialize result
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
            # Try alternative patterns
            if "error" not in response.lower():
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
            
            # Continue explanation if we're in one
            if current_step is not None and line and not line.startswith('Error Step Number:'):
                if line.lower().startswith('explanation:'):
                    current_explanation.append(line[12:].strip())  # Remove "explanation:"
                elif current_explanation:  # Continue multi-line explanation
                    current_explanation.append(line)
        
        # Don't forget the last step
        if current_step is not None and current_explanation:
            result['explanations'][current_step] = ' '.join(current_explanation).strip()
        
        # Remove duplicates and sort
        result['error_steps'] = sorted(list(set(result['error_steps'])))
        
        return result
    
    def batch_evaluate(self, examples: List[Dict[str, Any]], 
                      save_results: bool = True) -> List[CriticResult]:
        """Evaluate multiple examples in batch."""
        
        results = []
        
        print(f"Starting batch evaluation of {len(examples)} examples...")
        
        for i, example in enumerate(examples):
            print(f"Evaluating example {i+1}/{len(examples)}...")
            
            # Extract problem and steps
            problem = example.get('original_problem', {}).get('problem', '')
            
            # Use modified solution steps if available, otherwise original
            if 'modified_solution' in example and 'steps' in example['modified_solution']:
                steps = example['modified_solution']['steps']
            else:
                steps = example.get('original_problem', {}).get('parsed_steps', [])
            
            # Evaluate
            result = self.evaluate_solution(problem, steps)
            results.append(result)
            
            if result.has_errors:
                print(f"  ✓ Found {len(result.error_steps)} errors in steps: {result.error_steps}")
            else:
                print(f"  - No errors detected")
        
        if save_results:
            # Save results
            timestamp = int(time.time())
            filename = f"./data/critic_results_{timestamp}.json"
            
            serializable_results = [result.to_dict() for result in results]
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Results saved to {filename}")
        
        success_rate = sum(1 for r in results if r.has_errors) / len(results)
        print(f"Evaluation complete. Error detection rate: {success_rate:.1%}")
        
        return results


# Convenience function for quick testing
def evaluate_single_example(example: Dict[str, Any], model: str = "gpt-4o-mini") -> CriticResult:
    """Quick evaluation of a single example."""
    
    critic = LLMCritic(model=model)
    
    problem = example.get('original_problem', {}).get('problem', '')
    
    # Use modified solution if available
    if 'modified_solution' in example and 'steps' in example['modified_solution']:
        steps = example['modified_solution']['steps']
    else:
        steps = example.get('original_problem', {}).get('parsed_steps', [])
    
    return critic.evaluate_solution(problem, steps)