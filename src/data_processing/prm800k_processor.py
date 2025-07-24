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
            
            error_injection = LateBenchErrorInjection(
                has_errors=any(step.is_error for step in solution_steps),
                manual_attempts=[]
            )
            
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
                step_number=len(steps) + 1,  # Use sequential numbering instead of raw step index
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
                       max_examples: Optional[int] = None,
                       separate_by_errors: bool = True,
                       min_difficulty_level: Optional[int] = None,
                       min_solution_steps: Optional[int] = None) -> List[LateBenchExample]:
        """Process PRM800K dataset with advanced filtering and optional separation by error status"""
        
        print(f"Processing PRM800K dataset from {input_file}")
        if min_difficulty_level:
            print(f"Filtering for difficulty level >= {min_difficulty_level} (using solution complexity)")
        if min_solution_steps:
            print(f"Filtering for late error injection:")
            print(f"   • Complete solutions: >= {min_solution_steps} total steps")
            print(f"   • Error solutions: first error at step >= {min_solution_steps}")
        
        # Load JSONL data
        raw_data = []
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    raw_data.append(json.loads(line))
        
        print(f"Loaded {len(raw_data)} raw examples")
        
        # Process examples with filtering
        processed_examples = []
        complete_solutions = []
        error_labeled = []
        skipped = 0
        filtered_difficulty = 0
        filtered_steps = 0
        
        for i, raw_example in enumerate(raw_data):
            if max_examples and len(processed_examples) >= max_examples:
                break
            
            # Apply pre-processing filters
            if not self._passes_filters(raw_example, min_difficulty_level, min_solution_steps):
                if min_difficulty_level and not self._meets_difficulty_threshold(raw_example, min_difficulty_level):
                    filtered_difficulty += 1
                if min_solution_steps and not self._meets_step_threshold(raw_example, min_solution_steps):
                    filtered_steps += 1
                skipped += 1
                continue
                
            processed = self.process_example(raw_example, i)
            if processed:
                processed_examples.append(processed)
                
                # Separate by error status if requested
                if separate_by_errors:
                    if processed.source.metadata.get('has_errors', False):
                        error_labeled.append(processed)
                    else:
                        complete_solutions.append(processed)
                
                if len(processed_examples) % 100 == 0:
                    print(f"Processed {len(processed_examples)} examples")
            else:
                skipped += 1
        
        # Save results
        if separate_by_errors:
            # Save separate files for complete solutions and error-labeled problems
            base_path = output_file.replace('.json', '')
            complete_file = f"{base_path}_complete.json"
            error_file = f"{base_path}_errors.json"
            
            self.save_to_json(complete_solutions, complete_file)
            self.save_to_json(error_labeled, error_file)
            
            print(f"✅ Separated datasets:")
            print(f"   Complete solutions: {len(complete_solutions)} examples -> {complete_file}")
            print(f"   Error-labeled problems: {len(error_labeled)} examples -> {error_file}")
        
        # Still save the combined file
        self.save_to_json(processed_examples, output_file)
        
        # Print detailed filtering statistics
        print(f"✅ Completed: {len(processed_examples)} examples processed, {skipped} skipped")
        if filtered_difficulty > 0:
            print(f"   Filtered by difficulty: {filtered_difficulty} examples")
        if filtered_steps > 0:
            print(f"   Filtered by step count: {filtered_steps} examples")
        
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
                    "has_errors": example.error_injection.has_errors,
                    "error_info": example.error_injection.error_info,
                    "manual_attempts": [attempt.to_dict() for attempt in example.error_injection.manual_attempts],
                    "final_decision": example.error_injection.final_decision,
                    "decision_timestamp": example.error_injection.decision_timestamp,
                    "custom_suggestions": example.error_injection.custom_suggestions
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
    
    def _passes_filters(self, raw_example: Dict[str, Any], min_difficulty: Optional[int], min_steps: Optional[int]) -> bool:
        """Check if example passes all filtering criteria"""
        if min_difficulty and not self._meets_difficulty_threshold(raw_example, min_difficulty):
            return False
        if min_steps and not self._meets_step_threshold(raw_example, min_steps):
            return False
        return True
    
    def _meets_difficulty_threshold(self, raw_example: Dict[str, Any], min_level: int) -> bool:
        """Estimate difficulty based on solution complexity and mathematical content"""
        try:
            question = raw_example.get('question', {})
            problem = question.get('problem', '')
            solution = question.get('ground_truth_solution', '')
            
            # Calculate complexity score (1-5 scale)
            complexity_score = 1
            
            # Content-based difficulty indicators
            high_level_topics = [
                'integration', 'derivative', 'limit', 'differential', 'calculus',
                'matrix', 'eigenvalue', 'determinant', 'linear algebra',
                'topology', 'complex analysis', 'real analysis',
                'modular arithmetic', 'congruence', 'number theory',
                'combinatorics', 'probability', 'statistics'
            ]
            
            medium_level_topics = [
                'polynomial', 'quadratic', 'logarithm', 'exponential',
                'trigonometry', 'sine', 'cosine', 'tangent',
                'geometry', 'triangle', 'circle', 'coordinate',
                'sequence', 'series', 'function'
            ]
            
            problem_lower = problem.lower()
            
            # Base score from topic complexity
            if any(topic in problem_lower for topic in high_level_topics):
                complexity_score += 2
            elif any(topic in problem_lower for topic in medium_level_topics):
                complexity_score += 1
            
            # Solution length indicator
            solution_length = len(solution.split())
            if solution_length > 200:
                complexity_score += 2
            elif solution_length > 100:
                complexity_score += 1
            
            # Mathematical notation complexity
            complex_notation = ['\\int', '\\sum', '\\prod', '\\lim', '\\frac', '\\sqrt', '\\partial']
            notation_count = sum(1 for notation in complex_notation if notation in solution)
            if notation_count > 5:
                complexity_score += 1
            
            # Cap at 5
            complexity_score = min(5, complexity_score)
            
            return complexity_score >= min_level
            
        except Exception:
            return False
    
    def _meets_step_threshold(self, raw_example: Dict[str, Any], min_steps: int) -> bool:
        """Check if solution has enough steps for late error injection
        
        For complete solutions: must have >= min_steps total
        For error solutions: first error must be at step >= min_steps
        """
        try:
            question = raw_example.get('question', {})
            label = raw_example.get('label', {})
            
            # Get human annotation steps
            annotation_steps = label.get('steps', [])
            if not annotation_steps:
                return False
            
            # Find first error step (rating = -1)
            first_error_step = None
            for i, step in enumerate(annotation_steps):
                rating = step.get('completions', [{}])[0].get('rating', 1)
                if rating == -1:
                    first_error_step = i + 1  # Convert to 1-based indexing
                    break
            
            if first_error_step is None:
                # Complete solution - check total steps
                return len(annotation_steps) >= min_steps
            else:
                # Error solution - first error must be at step >= min_steps  
                return first_error_step >= min_steps
            
        except Exception as e:
            print(f"Error in step threshold check: {e}")
            return False
    
    def _get_first_error_step(self, raw_example: Dict[str, Any]) -> Optional[int]:
        """Get the position of the first error step (1-based indexing)"""
        try:
            label = raw_example.get('label', {})
            annotation_steps = label.get('steps', [])
            
            for i, step in enumerate(annotation_steps):
                rating = step.get('completions', [{}])[0].get('rating', 1)
                if rating == -1:
                    return i + 1  # Convert to 1-based indexing
            
            return None  # No error found (complete solution)
            
        except Exception:
            return None