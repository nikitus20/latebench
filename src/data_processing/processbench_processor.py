"""
ProcessBench Dataset Processor
Converts ProcessBench OlympiadBench data to unified LateBench format
Preserves all original data while extracting error step information
"""

import json
import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset

from unified_schema import (
    LateBenchExample, LateBenchSource, LateBenchProblem, 
    LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
    LateBenchProcessing, generate_latebench_id, create_timestamp
)


class ProcessBenchProcessor:
    """Pure converter: ProcessBench datasets (OlympiadBench and OmniMath) to unified LateBench format"""
    
    def __init__(self, split_name: str = "olympiadbench"):
        """
        Initialize converter for specific ProcessBench split
        
        Args:
            split_name: ProcessBench split ('olympiadbench' or 'omnimath')
        """
        self.split_name = split_name
        self.dataset_name = f"processbench_{split_name}"
        
    def process_example(self, raw_example: Dict[str, Any], index: int) -> Optional[LateBenchExample]:
        """Convert single ProcessBench example to LateBench format"""
        
        try:
            # Extract basic data with complete preservation
            original_id = raw_example.get('id', f"processbench_olympiad_{index}")
            problem_text = raw_example.get('problem', '').strip()
            steps_list = raw_example.get('steps', [])
            final_answer_correct = raw_example.get('final_answer_correct', True)
            label = raw_example.get('label', 0)  # First error step position
            generator = raw_example.get('generator', 'unknown')
            
            # Validate required data
            if not problem_text or not steps_list:
                return None
            
            # Parse solution steps with error information
            solution_steps = self._parse_processbench_steps(steps_list, label)
            if not solution_steps:
                return None
            
            # Create LateBench components with minimal assumptions
            lb_id = generate_latebench_id(self.dataset_name, original_id)
            
            # Preserve all original data in metadata
            # Determine competition type based on split
            competition_name = {
                "olympiadbench": "Mathematics Olympiad",
                "omnimath": "OmniMath Competition",
                "math": "MATH Competition", 
                "gsm8k": "GSM8K"
            }.get(self.split_name, f"ProcessBench {self.split_name}")
            
            source = LateBenchSource(
                dataset=self.dataset_name,
                original_id=original_id,
                difficulty=3,  # Default difficulty
                subject="mathematics",  # Generic default
                competition=competition_name,
                metadata={
                    # Preserve ALL original ProcessBench fields
                    "processbench_id": raw_example.get('id'),
                    "generator": generator,
                    "final_answer_correct": final_answer_correct,
                    "processbench_label": label,  # Original error step position
                    "has_errors": not final_answer_correct,  # Derived from correctness
                    "first_error_step": label if label > 0 else None,
                    "total_steps": len(steps_list),
                    # Store any additional fields that might exist
                    **{k: v for k, v in raw_example.items() 
                       if k not in ['id', 'problem', 'steps', 'final_answer_correct', 'label', 'generator']}
                }
            )
            
            problem = LateBenchProblem(statement=problem_text)
            
            # Extract final answer from last step if possible
            final_answer = self._extract_final_answer(steps_list)
            
            solution = LateBenchSolution(
                steps=solution_steps,
                final_answer=final_answer,
                total_steps=len(solution_steps),
                solution_method="analytical"
            )
            
            # Error injection info based on ProcessBench annotations
            has_errors = not final_answer_correct
            error_injection = LateBenchErrorInjection(
                has_errors=has_errors
            )
            
            processing = LateBenchProcessing(
                added_to_latebench=create_timestamp(),
                last_modified=create_timestamp(),
                status="processed",
                processor_version="1.0"
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
            print(f"Error processing ProcessBench example {index}: {e}")
            return None
    
    def _parse_processbench_steps(self, steps_list: List[str], first_error_step: int) -> List[LateBenchStep]:
        """Convert ProcessBench steps to LateBenchStep format with error information"""
        
        latebench_steps = []
        
        for i, step_content in enumerate(steps_list):
            step_number = i + 1  # 1-based indexing
            
            # Determine if this step contains an error
            is_error = (first_error_step > 0) and (step_number == first_error_step)
            
            # Classify importance based on error status
            if is_error:
                importance = "high"  # Error steps are crucial
            else:
                importance = "medium"  # Default for other steps
            
            # Basic reasoning type classification
            reasoning_type = self._classify_reasoning_type(step_content)
            
            latebench_steps.append(LateBenchStep(
                step_number=step_number,
                content=step_content.strip(),
                importance=importance,
                reasoning_type=reasoning_type,
                is_modified=False,  # These are original steps, not modified
                is_error=is_error
            ))
        
        return latebench_steps
    
    def _classify_reasoning_type(self, content: str) -> str:
        """Basic reasoning type classification from step content"""
        content_lower = content.lower()
        
        # Simple keyword-based classification
        if any(word in content_lower for word in ['calculate', 'compute', '=', '+', '-', '*', '/']):
            return "calculation"
        elif any(word in content_lower for word in ['substitute', 'solve', 'equation']):
            return "algebraic"
        elif any(word in content_lower for word in ['therefore', 'thus', 'since', 'because', 'conclude']):
            return "logical"
        elif any(word in content_lower for word in ['triangle', 'angle', 'geometric', 'distance']):
            return "geometric"
        else:
            return "analytical"
    
    def _extract_final_answer(self, steps_list: List[str]) -> str:
        """Extract final answer from the last step if boxed answer exists"""
        if not steps_list:
            return ""
        
        last_step = steps_list[-1]
        
        # Look for boxed answers
        import re
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, last_step)
        
        if match:
            return match.group(1)
        
        # Fallback: return last meaningful content
        lines = last_step.strip().split('\n')
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith('\\['):
                return line.strip()[:100]  # Truncate if too long
        
        return ""
    
    def convert_dataset(self, output_file: str) -> List[LateBenchExample]:
        """Pure conversion: Convert ProcessBench dataset to LateBench JSON format"""
        
        print(f"Converting ProcessBench {self.split_name} dataset...")
        
        try:
            # Load the dataset split
            dataset = load_dataset('Qwen/ProcessBench', split=self.split_name)
            print(f"Loaded {len(dataset)} raw examples from ProcessBench {self.split_name}")
            
        except Exception as e:
            print(f"❌ Error loading ProcessBench dataset: {e}")
            return []
        
        # Convert all examples
        converted_examples = []
        skipped = 0
        
        for i, raw_example in enumerate(dataset):
            converted = self.process_example(raw_example, i)
            if converted:
                converted_examples.append(converted)
                if len(converted_examples) % 50 == 0:
                    print(f"Converted {len(converted_examples)} examples")
            else:
                skipped += 1
        
        # Save results
        self.save_to_json(converted_examples, output_file)
        
        print(f"✅ Conversion completed: {len(converted_examples)} examples converted, {skipped} skipped")
        return converted_examples
    
    def save_to_json(self, examples: List[LateBenchExample], output_file: str):
        """Save LateBenchExample objects to JSON format using unified schema"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Use the to_dict() method from unified schema for consistent serialization
        json_data = [example.to_dict() for example in examples]
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved {len(examples)} examples to {output_file}")