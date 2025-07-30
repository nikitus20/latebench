"""
NuminaMath Dataset Processor
Converts NuminaMath data to unified LateBench format
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from .unified_schema import (
    LateBenchExample, LateBenchSource, LateBenchProblem, 
    LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
    LateBenchProcessing, generate_latebench_id, create_timestamp
)


class NuminaMathProcessor:
    """Process NuminaMath dataset into unified LateBench format"""
    
    def __init__(self):
        self.dataset_name = "numinamath"
        self.processed_count = 0
        
    def parse_solution_steps(self, solution: str) -> List[LateBenchStep]:
        """Parse solution text into individual steps"""
        if not solution or not isinstance(solution, str):
            return []
        
        steps = []
        lines = solution.strip().split('\n')
        current_step_content = []
        step_number = 1
        
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
            
            if is_new_step and current_step_content:
                # Save previous step
                content = ' '.join(current_step_content)
                steps.append(LateBenchStep(
                    step_number=step_number,
                    content=content,
                    importance="medium",  # NuminaMath doesn't have importance
                    reasoning_type=self._classify_reasoning_type(content)
                ))
                step_number += 1
                current_step_content = [line]
            elif is_new_step:
                current_step_content = [line]
            else:
                current_step_content.append(line)
        
        # Don't forget the last step
        if current_step_content:
            content = ' '.join(current_step_content)
            steps.append(LateBenchStep(
                step_number=step_number,
                content=content,
                importance="medium",
                reasoning_type=self._classify_reasoning_type(content)
            ))
        
        # If no clear step structure found, split by sentences/paragraphs
        if len(steps) <= 1:
            # Split by double newlines (paragraphs)
            paragraphs = solution.split('\n\n')
            if len(paragraphs) > 1:
                steps = []
                for i, paragraph in enumerate(paragraphs, 1):
                    paragraph = paragraph.strip()
                    if paragraph:
                        steps.append(LateBenchStep(
                            step_number=i,
                            content=paragraph,
                            importance="medium",
                            reasoning_type=self._classify_reasoning_type(paragraph)
                        ))
            else:
                # Split by sentences for very long single paragraphs
                sentences = re.split(r'[.!?]+', solution)
                steps = []
                for i, sentence in enumerate(sentences, 1):
                    sentence = sentence.strip()
                    if sentence and len(sentence) > 10:  # Skip very short fragments
                        steps.append(LateBenchStep(
                            step_number=i,
                            content=sentence,
                            importance="medium",
                            reasoning_type=self._classify_reasoning_type(sentence)
                        ))
        
        return [step for step in steps if step.content.strip()]
    
    def _classify_reasoning_type(self, content: str) -> str:
        """Classify the type of reasoning in a step"""
        content_lower = content.lower()
        
        # Look for calculation indicators
        if any(indicator in content_lower for indicator in ['calculate', 'compute', '=', 'multiply', 'divide', 'add', 'subtract']):
            return "calculation"
        
        # Look for algebraic manipulation
        if any(indicator in content_lower for indicator in ['solve', 'equation', 'substitute', 'factor', 'expand']):
            return "algebraic"
        
        # Look for geometric reasoning
        if any(indicator in content_lower for indicator in ['triangle', 'angle', 'area', 'perimeter', 'circle', 'square']):
            return "geometric"
        
        # Look for logical reasoning
        if any(indicator in content_lower for indicator in ['therefore', 'thus', 'since', 'because', 'implies']):
            return "logical"
        
        # Look for theorem application
        if any(indicator in content_lower for indicator in ['theorem', 'law', 'formula', 'property', 'rule']):
            return "theorem_application"
        
        # Look for case analysis
        if any(indicator in content_lower for indicator in ['case', 'when', 'if', 'either', 'consider']):
            return "case_analysis"
        
        return "unknown"
    
    def _extract_subject(self, problem_text: str) -> str:
        """Extract subject area from problem text"""
        problem_lower = problem_text.lower()
        
        if any(term in problem_lower for term in ['triangle', 'circle', 'angle', 'area', 'perimeter', 'geometric']):
            return "geometry"
        elif any(term in problem_lower for term in ['integral', 'derivative', 'limit', 'calculus']):
            return "calculus"
        elif any(term in problem_lower for term in ['probability', 'random', 'expected', 'variance']):
            return "probability"
        elif any(term in problem_lower for term in ['matrix', 'vector', 'linear', 'determinant']):
            return "linear_algebra"
        elif any(term in problem_lower for term in ['sequence', 'series', 'recursive', 'fibonacci']):
            return "sequences"
        elif any(term in problem_lower for term in ['polynomial', 'equation', 'variable', 'solve']):
            return "algebra"
        elif any(term in problem_lower for term in ['prime', 'divisible', 'integer', 'modular']):
            return "number_theory"
        else:
            return "mathematics"
    
    def process_example(self, raw_example: Dict[str, Any], index: int) -> Optional[LateBenchExample]:
        """Convert single NuminaMath example to LateBench format"""
        try:
            # Check if this is already processed data (error injection results)
            if 'original_problem' in raw_example:
                original_problem = raw_example['original_problem']
                original_id = f"numina_processed_{index}"
                problem_text = original_problem.get('problem', '')
                solution_text = original_problem.get('solution', '')
                answer = original_problem.get('answer', 'No answer provided')
                source = original_problem.get('source', 'numinamath')
                parsed_steps = original_problem.get('parsed_steps', [])
            else:
                # Raw NuminaMath format
                original_id = raw_example.get('original_id', f"numina_{index}")
                problem_text = raw_example.get('problem', '')
                solution_text = raw_example.get('solution', '')
                answer = raw_example.get('answer', 'No answer provided')
                source = raw_example.get('source', 'numinamath')
                parsed_steps = []
            
            if not problem_text:
                return None
            
            # Generate LateBench ID
            lb_id = generate_latebench_id(self.dataset_name, original_id)
            
            # Create source info
            source = LateBenchSource(
                dataset=self.dataset_name,
                original_id=original_id,
                difficulty=3.0,  # Default medium
                subject=self._extract_subject(problem_text),
                competition=raw_example.get('source', '').split('_')[0] if raw_example.get('source') else None,
                metadata=raw_example.get('metadata', {})
            )
            
            # Create problem
            problem = LateBenchProblem(statement=problem_text)
            
            # Parse solution into steps - use pre-parsed steps if available
            if parsed_steps:
                steps = []
                for i, step_text in enumerate(parsed_steps, 1):
                    steps.append(LateBenchStep(
                        step_number=i,
                        content=step_text,
                        importance="medium",
                        reasoning_type=self._classify_reasoning_type(step_text)
                    ))
            else:
                steps = self.parse_solution_steps(solution_text)
            
            if not steps:
                return None
                
            solution = LateBenchSolution(
                steps=steps,
                final_answer=answer,
                total_steps=len(steps),
                solution_method="analytical"
            )
            
            # Initialize error injection data
            error_injection = LateBenchErrorInjection()
            
            # Processing info
            processing = LateBenchProcessing(
                added_to_latebench=create_timestamp(),
                last_modified=create_timestamp(),
                status="processed"
            )
            
            return LateBenchExample(
                id=lb_id,
                source=source,
                problem=problem,
                solution=solution,
                error_injection=error_injection,
                processing=processing
            )
            
        except Exception as e:
            print(f"Error processing NuminaMath example {index}: {e}")
            return None
    
    def process_dataset(self, 
                       input_file: str, 
                       output_file: str,
                       max_examples: Optional[int] = None) -> List[LateBenchExample]:
        """Process complete NuminaMath dataset"""
        
        print(f"Processing NuminaMath dataset from {input_file}")
        
        # Load raw data
        with open(input_file, 'r') as f:
            raw_data = json.load(f)
        
        if max_examples:
            raw_data = raw_data[:max_examples]
        
        processed_examples = []
        skipped = 0
        
        for i, raw_example in enumerate(raw_data):
            example = self.process_example(raw_example, i)
            if example:
                processed_examples.append(example)
                self.processed_count += 1
            else:
                skipped += 1
                
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(raw_data)} examples ({skipped} skipped)")
        
        # Save processed data
        print(f"Saving {len(processed_examples)} examples to {output_file}")
        with open(output_file, 'w') as f:
            json.dump([example.to_dict() for example in processed_examples], f, indent=2)
        
        print(f"✅ NuminaMath processing complete: {self.processed_count} examples processed, {skipped} skipped")
        
        return processed_examples


def main():
    """Main processing function for NuminaMath"""
    processor = NuminaMathProcessor()
    
    # Process the existing educational examples
    input_file = "./data/educational_examples.json"
    output_file = "./data/datasets/latebench_numinamath_raw.json"
    
    if os.path.exists(input_file):
        processor.process_dataset(input_file, output_file)
    else:
        print(f"Input file not found: {input_file}")


if __name__ == "__main__":
    main()