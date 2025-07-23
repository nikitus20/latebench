"""
PRM800K Dataset Processor - Clean and Simple
Converts PRM800K Phase 2 data to unified LateBench format
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from .unified_schema import (
    LateBenchExample, LateBenchSource, LateBenchProblem, 
    LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
    LateBenchProcessing, generate_latebench_id, create_timestamp,
    normalize_difficulty
)


class PRM800KProcessor:
    """Process PRM800K Phase 2 dataset into unified LateBench format"""
    
    def __init__(self):
        self.dataset_name = "prm800k"
        self.processed_count = 0
        
    def process_example(self, raw_example: Dict[str, Any], index: int) -> Optional[LateBenchExample]:
        """Convert single PRM800K Phase 2 example to LateBench format"""
        
        # Validate Phase 2 format
        if not self._is_valid_phase2_format(raw_example):
            return None
        
        try:
            question = raw_example['question']
            label = raw_example['label']
            
            # Extract basic data
            original_id = question.get('unique_id', f"prm_phase2_{index}")
            problem_text = question.get('problem', '').strip()
            answer = question.get('ground_truth_answer', '').strip()
            finish_reason = label.get('finish_reason', '')
            
            # Validate required data
            if not problem_text:
                return None
            
            # Extract human annotation steps
            solution_steps = self._parse_human_annotations(label)
            if not solution_steps:
                return None
            
            # Create LateBench components
            lb_id = generate_latebench_id(self.dataset_name, original_id)
            
            source = LateBenchSource(
                dataset=self.dataset_name,
                original_id=original_id,
                difficulty=normalize_difficulty(3, self.dataset_name),
                subject=self._classify_subject(problem_text),
                competition="Math Competition",
                metadata={
                    "finish_reason": finish_reason,
                    "human_steps": len(solution_steps),
                    "has_errors": any(step.is_error for step in solution_steps)
                }
            )
            
            problem = LateBenchProblem(statement=problem_text)
            
            solution = LateBenchSolution(
                steps=solution_steps,
                final_answer=answer,
                total_steps=len(solution_steps),
                solution_method="analytical"
            )
            
            error_injection = LateBenchErrorInjection()
            
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
            print(f"Error processing PRM800K example {index}: {e}")
            return None
    
    def _is_valid_phase2_format(self, raw_example: Dict[str, Any]) -> bool:
        """Validate Phase 2 format"""
        return (
            'question' in raw_example and 
            'label' in raw_example and
            'problem' in raw_example['question'] and
            'steps' in raw_example['label']
        )
    
    def _parse_human_annotations(self, label: Dict[str, Any]) -> List[LateBenchStep]:
        """Parse human annotation steps with proper error detection"""
        
        steps = []
        first_error_found = False
        
        for i, step_data in enumerate(label.get('steps', [])):
            if 'completions' not in step_data or not step_data['completions']:
                continue
                
            completion = step_data['completions'][0]
            step_text = completion.get('text', '').strip()
            rating = completion.get('rating', 0)
            
            if not step_text:
                continue
            
            # Determine step properties from human rating
            is_error = (rating == -1)
            is_first_error = is_error and not first_error_found
            if is_first_error:
                first_error_found = True
            
            importance = self._get_importance_from_rating(rating)
            reasoning_type = self._classify_reasoning_type(step_text)
            
            steps.append(LateBenchStep(
                step_number=i + 1,
                content=step_text,
                importance=importance,
                reasoning_type=reasoning_type,
                is_error=is_error,
                is_modified=is_first_error  # Mark first error as modified
            ))
        
        return steps
    
    def _get_importance_from_rating(self, rating: int) -> str:
        """Get step importance from human rating"""
        if rating == -1:
            return "high"  # Error steps are important
        elif rating == 1:
            return "medium"  # Correct steps
        else:  # rating == 0
            return "low"  # Questionable steps
    
    def _classify_reasoning_type(self, content: str) -> str:
        """Classify reasoning type"""
        content_lower = content.lower()
        
        if re.search(r'[+\-*/=]|calculate|compute', content_lower):
            return "calculation"
        elif any(term in content_lower for term in ['solve', 'equation', 'substitute']):
            return "algebraic"
        elif any(term in content_lower for term in ['therefore', 'thus', 'since', 'because']):
            return "logical"
        else:
            return "analytical"
    
    def _classify_subject(self, problem_text: str) -> str:
        """Classify subject from problem text"""
        problem_lower = problem_text.lower()
        
        if any(term in problem_lower for term in ['triangle', 'circle', 'angle', 'geometric']):
            return "geometry"
        elif any(term in problem_lower for term in ['equation', 'polynomial', 'function']):
            return "algebra"
        elif any(term in problem_lower for term in ['probability', 'combinatorics']):
            return "combinatorics"
        else:
            return "mathematics"
    
    def process_dataset(self, 
                       input_file: str, 
                       output_file: str,
                       max_examples: Optional[int] = None) -> List[LateBenchExample]:
        """Process PRM800K dataset"""
        
        print(f"Processing PRM800K dataset from {input_file}")
        
        # Load JSONL data
        raw_data = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        
        print(f"Loaded {len(raw_data)} raw examples")
        
        # Process examples
        processed_examples = []
        skipped = 0
        
        for i, raw_example in enumerate(raw_data):
            if max_examples and len(processed_examples) >= max_examples:
                break
                
            processed = self.process_example(raw_example, i)
            if processed:
                processed_examples.append(processed)
                if len(processed_examples) % 100 == 0:
                    print(f"Processed {len(processed_examples)} examples")
            else:
                skipped += 1
        
        # Save results
        self.save_to_json(processed_examples, output_file)
        
        print(f"✅ Completed: {len(processed_examples)} examples processed, {skipped} skipped")
        return processed_examples
    
    def save_to_json(self, examples: List[LateBenchExample], output_file: str):
        """Save to JSON format"""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        json_data = []
        for example in examples:
            json_data.append({
                "id": example.id,
                "source": {
                    "dataset": example.source.dataset,
                    "original_id": example.source.original_id,
                    "difficulty": example.source.difficulty,
                    "subject": example.source.subject,
                    "competition": example.source.competition,
                    "metadata": example.source.metadata
                },
                "problem": {
                    "statement": example.problem.statement
                },
                "solution": {
                    "steps": [
                        {
                            "step_number": step.step_number,
                            "content": step.content,
                            "importance": step.importance,
                            "reasoning_type": step.reasoning_type,
                            "is_error": step.is_error,
                            "is_modified": step.is_modified
                        } for step in example.solution.steps
                    ],
                    "final_answer": example.solution.final_answer,
                    "total_steps": example.solution.total_steps,
                    "solution_method": example.solution.solution_method
                },
                "error_injection": {
                    "has_errors": example.error_injection.has_errors
                },
                "processing": {
                    "added_to_latebench": example.processing.added_to_latebench,
                    "last_modified": example.processing.last_modified,
                    "status": example.processing.status
                }
            })
        
        with open(output_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved {len(examples)} examples to {output_file}")