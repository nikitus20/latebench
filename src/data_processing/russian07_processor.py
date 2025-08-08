"""
Russian07 Dataset Processor
Converts Russian07 olympiad mathematics dataset to unified LateBench format
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


class Russian07Processor:
    """Process Russian07 olympiad mathematics dataset into unified LateBench format"""
    
    def __init__(self):
        self.dataset_name = "russian07"
        self.processed_count = 0
        self.successful_conversions = 0
        
    def parse_solution_steps(self, solution_text: str) -> List[LateBenchStep]:
        """Parse solution text into individual steps"""
        if not solution_text:
            return []
        
        steps = []
        
        # Split by "Шаг" (Step) markers
        step_pattern = r'Шаг\s+(\d+):\s*(.*?)(?=Шаг\s+\d+:|Ответ:|$)'
        matches = re.findall(step_pattern, solution_text, re.DOTALL | re.IGNORECASE)
        
        if not matches:
            # Fallback: try to split by numbered patterns like "1.", "2.", etc.
            step_pattern = r'(\d+)\.\s*(.*?)(?=\d+\.|Ответ:|$)'
            matches = re.findall(step_pattern, solution_text, re.DOTALL)
        
        if not matches:
            # If no clear step structure, treat as single step
            steps.append(LateBenchStep(
                step_number=1,
                content=solution_text.strip(),
                importance="medium",
                reasoning_type="unknown"
            ))
            return steps
        
        for i, (step_num, content) in enumerate(matches):
            clean_content = content.strip()
            if clean_content:
                steps.append(LateBenchStep(
                    step_number=int(step_num),
                    content=clean_content,
                    importance="medium",  # Default importance
                    reasoning_type=self._classify_reasoning_type(clean_content)
                ))
        
        return steps
    
    def _classify_reasoning_type(self, content: str) -> str:
        """Classify the type of reasoning in a step"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['обозначим', 'пусть', 'рассмотрим']):
            return "setup"
        elif any(word in content_lower for word in ['тождество', 'формула', 'равенство']):
            return "algebraic"
        elif any(word in content_lower for word in ['неравенство', 'больше', 'меньше']):
            return "inequality"
        elif any(word in content_lower for word in ['делится', 'остаток', 'простое']):
            return "number_theory"
        elif any(word in content_lower for word in ['геометри', 'треугольник', 'окружность']):
            return "geometric"
        elif any(word in content_lower for word in ['следовательно', 'поэтому', 'значит']):
            return "logical"
        elif any(word in content_lower for word in ['проверка', 'подставим']):
            return "verification"
        else:
            return "unknown"
    
    def extract_final_answer(self, solution_text: str) -> str:
        """Extract final answer from solution text"""
        # Look for "Ответ:" pattern
        answer_match = re.search(r'Ответ:\s*(.*?)(?:\n|$)', solution_text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: look for the last line if it seems like an answer
        lines = solution_text.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            if len(last_line) < 200:  # Reasonable answer length
                return last_line
        
        return "Не указан"
    
    def _determine_subject(self, statement: str, hint: str = "", solution: str = "") -> str:
        """Determine mathematical subject based on problem content"""
        content = f"{statement} {hint} {solution}".lower()
        
        if any(word in content for word in ['простое', 'делится', 'остаток', 'НОД', 'НОК']):
            return "number_theory"
        elif any(word in content for word in ['треугольник', 'окружность', 'геометри', 'биссектриса']):
            return "geometry"
        elif any(word in content for word in ['комбинатори', 'размещени', 'сочетани', 'способ']):
            return "combinatorics"
        elif any(word in content for word in ['граф', 'вершин', 'ребр']):
            return "graph_theory"
        elif any(word in content for word in ['игр', 'стратеги', 'выигрыш']):
            return "game_theory"
        elif any(word in content for word in ['алгебр', 'уравнени', 'неравенств']):
            return "algebra"
        else:
            return "mathematics"
    
    def process_problem(self, problem_data: Dict[str, Any]) -> Optional[LateBenchExample]:
        """Process a single Russian07 problem into LateBench format"""
        try:
            # Extract basic information
            problem_id = problem_data.get('id', '')
            statement = problem_data.get('statement', '')
            solution_text = problem_data.get('solution', '')
            hint = problem_data.get('hint', '')
            
            # Extract new complexity data
            complexity_beta = problem_data.get('complexity_beta')
            difficulty_category = problem_data.get('difficulty_category', 'medium')
            solver_percentage = problem_data.get('solver_percentage')
            
            if not statement or not solution_text:
                return None
            
            # Parse solution into steps
            solution_steps = self.parse_solution_steps(solution_text)
            if not solution_steps:
                return None
            
            # Extract final answer
            final_answer = self.extract_final_answer(solution_text)
            
            # Determine subject
            subject = self._determine_subject(statement, hint, solution_text)
            
            # Map difficulty category to numeric value
            difficulty_mapping = {
                'easy': 5,
                'medium': 7,
                'hard': 9
            }
            numeric_difficulty = difficulty_mapping.get(difficulty_category, 7)
            
            # Create LateBench components
            lb_id = generate_latebench_id("russian07", problem_id)
            
            source = LateBenchSource(
                dataset="russian07",
                original_id=problem_id,
                difficulty=numeric_difficulty,
                subject=subject,
                competition="Russian Mathematical Olympiad",
                year=None,  # Year not specified in dataset
                metadata={
                    "origin": problem_data.get('origin', ''),
                    "problem_number": problem_data.get('problem_number', ''),
                    "hint": hint,
                    "has_subparts": problem_data.get('has_subparts', False),
                    "subparts": problem_data.get('subparts', []),
                    "language": "Russian",
                    "complexity_beta": complexity_beta,
                    "difficulty_category": difficulty_category,
                    "solver_percentage": solver_percentage
                }
            )
            
            problem = LateBenchProblem(
                statement=statement,
                hint=hint if hint.strip() else None
            )
            
            solution = LateBenchSolution(
                steps=solution_steps,
                final_answer=final_answer,
                solution_method="analytical"
            )
            
            error_injection = LateBenchErrorInjection()
            
            processing = LateBenchProcessing(
                added_to_latebench=create_timestamp(),
                last_modified=create_timestamp(),
                status="processed",
                processor_version="1.0"
            )
            
            example = LateBenchExample(
                id=lb_id,
                source=source,
                problem=problem,
                solution=solution,
                error_injection=error_injection,
                processing=processing
            )
            
            self.successful_conversions += 1
            return example
            
        except Exception as e:
            print(f"❌ Error processing problem {problem_data.get('id', 'unknown')}: {e}")
            return None
    
    def process_dataset(self, input_file: str, output_file: str, max_examples: Optional[int] = None) -> List[LateBenchExample]:
        """Process entire Russian07 dataset"""
        print(f"🔄 Processing Russian07 dataset from {input_file}")
        
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        # Load dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        problems = dataset.get('problems', [])
        total_problems = len(problems)
        
        if max_examples:
            problems = problems[:max_examples]
            print(f"📊 Processing first {len(problems)} of {total_problems} problems")
        else:
            print(f"📊 Processing all {total_problems} problems")
        
        examples = []
        
        for i, problem_data in enumerate(problems):
            if (i + 1) % 50 == 0:
                print(f"📈 Processed {i + 1}/{len(problems)} problems...")
            
            self.processed_count += 1
            example = self.process_problem(problem_data)
            
            if example:
                examples.append(example)
        
        print(f"✅ Successfully converted {self.successful_conversions}/{self.processed_count} problems")
        
        # Save to output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([example.to_dict() for example in examples], f, 
                     indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(examples)} examples to {output_file}")
        
        return examples


def main():
    """Main function for command-line usage"""
    processor = Russian07Processor()
    
    input_file = "data/russian07_final.json"
    output_file = "data/datasets/latebench_russian07_final.json"
    
    try:
        examples = processor.process_dataset(input_file, output_file)
        print(f"🎉 Russian07 processing complete! Generated {len(examples)} LateBench examples")
        
        # Print some statistics
        subjects = {}
        step_counts = []
        
        for example in examples:
            subject = example.source.subject
            subjects[subject] = subjects.get(subject, 0) + 1
            step_counts.append(len(example.solution.steps))
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total examples: {len(examples)}")
        print(f"  Average steps per solution: {sum(step_counts)/len(step_counts):.1f}")
        print(f"  Min/Max steps: {min(step_counts)}/{max(step_counts)}")
        print(f"  Subjects: {dict(sorted(subjects.items()))}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()