"""
PRM800K Dataset Processor with Human Annotations
Converts PRM800K MATH dataset WITH step-level human ratings to unified LateBench format
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
try:
    from .unified_schema import (
        LateBenchExample, LateBenchSource, LateBenchProblem, 
        LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
        LateBenchProcessing, generate_latebench_id, create_timestamp
    )
except ImportError:
    from unified_schema import (
        LateBenchExample, LateBenchSource, LateBenchProblem, 
        LateBenchSolution, LateBenchStep, LateBenchErrorInjection,
        LateBenchProcessing, generate_latebench_id, create_timestamp
    )


class PRM800KProcessor:
    """Process PRM800K MATH dataset WITH human annotations into unified LateBench format"""
    
    def __init__(self):
        self.dataset_name = "prm800k"
        self.processed_count = 0
        self.errors_found = 0
        self.successful_reconstructions = 0
        
    def reconstruct_solution_trajectory(self, label_data: Dict[str, Any]) -> List[LateBenchStep]:
        """Reconstruct the actual solution trajectory from human annotations with step-level ratings"""
        if not label_data or 'steps' not in label_data:
            return []
        
        steps = []
        step_number = 1
        found_error = False
        finish_reason = label_data.get('finish_reason', '')
        
        for step_data in label_data['steps']:
            try:
                # Determine which completion was chosen
                chosen_completion = step_data.get('chosen_completion')
                human_completion = step_data.get('human_completion')
                
                # Initialize variables
                step_content = ""
                rating = 0
                is_error = False
                
                # Get the actual step content and rating
                if human_completion is not None:
                    # Human wrote their own step
                    step_content = human_completion
                    rating = 1  # Human completions are considered good
                    is_error = False
                elif chosen_completion is not None and 'completions' in step_data:
                    # Use the chosen completion
                    completions = step_data['completions']
                    if 0 <= chosen_completion < len(completions):
                        completion = completions[chosen_completion]
                        step_content = completion.get('text', '')
                        rating = completion.get('rating', 0)
                        is_error = (rating == -1)
                    else:
                        continue  # Skip invalid chosen_completion
                else:
                    # No completion was chosen - this often happens when annotator found errors
                    # If finish_reason is 'found_error', let's include the first error step we find
                    if finish_reason == 'found_error' and 'completions' in step_data:
                        completions = step_data['completions']
                        # Look for the first error completion (-1 rating)
                        error_completion = None
                        for comp in completions:
                            if comp.get('rating', 0) == -1:
                                error_completion = comp
                                break
                        
                        if error_completion:
                            step_content = error_completion.get('text', '')
                            rating = -1
                            is_error = True
                        else:
                            # No error found, skip this step
                            continue
                    else:
                        # No valid completion found and not an error case
                        continue
                
                if not step_content or not step_content.strip():
                    continue
                
                # Update found_error flag
                if is_error:
                    found_error = True
                
                # Map rating to importance
                importance = self._map_rating_to_importance(rating)
                
                # Create LateBench step with human annotations
                step = LateBenchStep(
                    step_number=step_number,
                    content=step_content.strip(),
                    importance=importance,
                    reasoning_type=self._classify_reasoning_type(step_content),
                    is_modified=False,  # Original human-annotated step
                    is_error=is_error
                )
                
                steps.append(step)
                step_number += 1
                
                # If we found an error and this is a 'found_error' case, stop here
                if finish_reason == 'found_error' and is_error:
                    break
                
            except Exception as e:
                print(f"Warning: Error processing step {step_number}: {e}")
                continue
        
        if found_error:
            self.errors_found += 1
        
        if steps:
            self.successful_reconstructions += 1
            
        return steps
    
    def _map_rating_to_importance(self, rating: int) -> str:
        """Map PRM800K human rating (-1, 0, +1) to LateBench importance"""
        if rating == -1:
            return "high"  # Error steps are high importance
        elif rating == 1:
            return "high"  # Good steps are high importance
        else:  # rating == 0
            return "medium"  # Neutral steps are medium importance
    
    def process_example(self, raw_example: Dict[str, Any], index: int) -> Optional[LateBenchExample]:
        """Process a single PRM800K example with human annotations into LateBench format"""
        try:
            # Extract question data
            question_data = raw_example.get('question', {})
            label_data = raw_example.get('label', {})
            
            problem_text = question_data.get('problem', '')
            ground_truth_solution = question_data.get('ground_truth_solution', '')
            ground_truth_answer = question_data.get('ground_truth_answer', '')
            
            if not problem_text:
                return None
            
            # Reconstruct solution from human annotations
            steps = self.reconstruct_solution_trajectory(label_data)
            if not steps:
                # Fallback: create steps from ground truth if trajectory reconstruction fails
                steps = self._create_fallback_steps(ground_truth_solution)
            
            # Extract original error steps from the reconstructed solution
            original_error_steps = [step.step_number for step in steps if step.is_error]
            
            # Determine if this example has errors
            finish_reason = label_data.get('finish_reason', '')
            has_natural_errors = (finish_reason == 'found_error')
            
            # Generate unique ID from labeler and timestamp for uniqueness
            labeler_id = raw_example.get('labeler', 'unknown')[:8]
            timestamp = raw_example.get('timestamp', '').replace(':', '').replace('-', '')[:12]
            unique_id = f"prm800k_{labeler_id}_{timestamp}_{index}"
            
            # Create LateBench components
            source = LateBenchSource(
                dataset=self.dataset_name,
                original_id=unique_id,
                difficulty=5,  # PRM800K problems are typically challenging
                subject="mathematics",  # Generic subject for now
                competition="MATH",
                metadata={
                    "dataset_index": index,
                    "original_source": "PRM800K",
                    "labeler_id": labeler_id,
                    "timestamp": raw_example.get('timestamp', ''),
                    "generation": raw_example.get('generation'),
                    "finish_reason": finish_reason,
                    "has_natural_errors": has_natural_errors,
                    "total_annotation_time_ms": label_data.get('total_time', 0),
                    "is_quality_control": raw_example.get('is_quality_control_question', False),
                    "human_annotated": True,
                    "step_count": len(steps)
                }
            )
            
            problem = LateBenchProblem(statement=problem_text)
            
            # Use reconstructed trajectory as the solution, not ground truth
            solution = LateBenchSolution(
                steps=steps,
                final_answer=ground_truth_answer,  # Keep ground truth answer for comparison
                solution_method="human_annotated"
            )
            
            # Set up error injection to indicate this has natural errors
            error_injection = LateBenchErrorInjection(
                has_errors=has_natural_errors
            )
            if has_natural_errors:
                error_injection.error_info = {
                    "error_type": "natural_human_identified",
                    "finish_reason": finish_reason,
                    "source": "PRM800K_human_annotation"
                }
            
            processing = LateBenchProcessing(
                added_to_latebench=create_timestamp(),
                last_modified=create_timestamp(),
                status="processed_with_annotations",
                processor_version="2.0_with_human_annotations"
            )
            
            # Generate unique LateBench ID
            latebench_id = generate_latebench_id(self.dataset_name, unique_id)
            
            example = LateBenchExample(
                id=latebench_id,
                source=source,
                problem=problem,
                solution=solution,
                error_injection=error_injection,
                processing=processing,
                original_error_steps=original_error_steps,  # Error steps from human annotations
                injected_error_steps=[]  # No error injection during PRM800K processing
            )
            
            self.processed_count += 1
            return example
            
        except Exception as e:
            print(f"Error processing example {index}: {e}")
            return None
    
    def _create_fallback_steps(self, ground_truth_solution: str) -> List[LateBenchStep]:
        """Create fallback steps from ground truth solution if trajectory reconstruction fails"""
        if not ground_truth_solution:
            return [LateBenchStep(
                step_number=1,
                content="No solution available",
                importance="medium",
                reasoning_type="unknown"
            )]
        
        # Simple sentence-based splitting for ground truth
        sentences = re.split(r'(?<=[.!?])\s+|\n+', ground_truth_solution.strip())
        steps = []
        step_number = 1
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Skip very short fragments
                steps.append(LateBenchStep(
                    step_number=step_number,
                    content=sentence,
                    importance="medium",
                    reasoning_type=self._classify_reasoning_type(sentence)
                ))
                step_number += 1
        
        return steps if steps else [LateBenchStep(
            step_number=1,
            content=ground_truth_solution,
            importance="medium", 
            reasoning_type="unknown"
        )]
    
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
        
        return "unknown"
    
    def process_dataset(self, input_file: str, output_file: str, max_examples: Optional[int] = None):
        """Process entire PRM800K dataset file with human annotations"""
        print(f"🔄 Processing PRM800K dataset with human annotations: {input_file}")
        
        examples = []
        processed_count = 0
        
        try:
            with open(input_file, 'r') as f:
                for i, line in enumerate(f):
                    if max_examples and processed_count >= max_examples:
                        break
                        
                    try:
                        raw_example = json.loads(line.strip())
                        example = self.process_example(raw_example, i)
                        if example:
                            examples.append(example)
                            processed_count += 1
                            
                        if processed_count % 100 == 0:
                            print(f"   Processed {processed_count} examples (errors found: {self.errors_found}, successful reconstructions: {self.successful_reconstructions})...")
                            
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error on line {i}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {i}: {e}")
                        continue
        
        except FileNotFoundError:
            print(f"❌ File not found: {input_file}")
            return []
        
        print(f"✅ Successfully processed {len(examples)} examples")
        print(f"📊 Statistics:")
        print(f"   - Examples with natural errors: {self.errors_found}")
        print(f"   - Successful trajectory reconstructions: {self.successful_reconstructions}")
        print(f"   - Total processed: {self.processed_count}")
        
        # Save to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump([example.to_dict() for example in examples], f, indent=2)
        
        print(f"💾 Saved to {output_file}")
        return examples


def main():
    """Convert PRM800K human-annotated data to LateBench format"""
    processor = PRM800KProcessor()
    
    # Process the human-annotated PRM800K files from prm800k_temp
    base_path = "prm800k_temp/prm800k/data"
    
    # Process phase 2 training data (contains the most annotations)
    print("🚀 Processing PRM800K Phase 2 Training Data with Human Annotations")
    processor.process_dataset(
        f"{base_path}/phase2_train.jsonl",
        "data/datasets/latebench_prm800k_complete.json"
        # No limit - process full dataset
    )
    
    print("🎯 PRM800K processing complete with human annotations!")
    print("🔍 The dataset now includes step-level human ratings and error identification!")

if __name__ == "__main__":
    main()