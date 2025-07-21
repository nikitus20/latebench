"""
Logical error taxonomy for mathematical reasoning.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


@dataclass
class ErrorType:
    """Definition of a logical error type."""
    name: str
    description: str
    example: str
    mathematical_context: str
    detection_difficulty: str
    common_subjects: List[str]


class LogicalErrorCategory(Enum):
    """Categories of logical errors in mathematical reasoning."""
    GENERALIZATION = "generalization"
    THEOREM_APPLICATION = "theorem_application" 
    CASE_ANALYSIS = "case_analysis"
    LOGICAL_STRUCTURE = "logical_structure"
    DOMAIN_OPERATIONS = "domain_operations"
    QUANTIFICATION = "quantification"
    SUBSTITUTION = "substitution"
    EQUIVALENCE = "equivalence"


class MathematicalErrorTaxonomy:
    """Complete taxonomy of logical errors for mathematical reasoning."""
    
    def __init__(self):
        self.error_types = self._define_error_types()
    
    def _define_error_types(self) -> Dict[str, ErrorType]:
        """Define all error types with detailed specifications."""
        
        return {
            "invalid_generalization": ErrorType(
                name="invalid_generalization",
                description="Applying a specific case result to general case without justification",
                example="Because the formula works for n=1,2,3, it works for all positive integers",
                mathematical_context="Often occurs in proofs by induction, series analysis, or pattern recognition",
                detection_difficulty="High - appears logical and follows pattern-based thinking",
                common_subjects=["algebra", "number_theory", "combinatorics", "analysis"]
            ),
            
            "theorem_misapplication": ErrorType(
                name="theorem_misapplication", 
                description="Using a theorem outside its valid conditions or domain",
                example="Applying L'Hôpital's rule when limit isn't indeterminate form",
                mathematical_context="Requires careful attention to theorem prerequisites and assumptions",
                detection_difficulty="Medium - requires knowledge of theorem conditions",
                common_subjects=["calculus", "analysis", "geometry", "algebra"]
            ),
            
            "incomplete_case_analysis": ErrorType(
                name="incomplete_case_analysis",
                description="Missing important cases in proof by cases or overlooking boundary conditions", 
                example="Only considering positive values when negative values are also valid",
                mathematical_context="Common in absolute value problems, piecewise functions, optimization",
                detection_difficulty="Medium - requires systematic verification of all cases",
                common_subjects=["algebra", "analysis", "geometry", "discrete_math"]
            ),
            
            "circular_reasoning": ErrorType(
                name="circular_reasoning",
                description="Using the conclusion to prove itself, directly or indirectly",
                example="Assuming P to prove P, or assuming P to prove Q then using Q to prove P",
                mathematical_context="Subtle in complex proofs where the circular dependency is indirect",
                detection_difficulty="High - circular dependency may span multiple steps",
                common_subjects=["logic", "algebra", "geometry", "analysis"]
            ),
            
            "false_equivalence": ErrorType(
                name="false_equivalence", 
                description="Treating non-equivalent statements as equivalent",
                example="Treating 'If A then B' as equivalent to 'If B then A'",
                mathematical_context="Common with logical implications, set relationships, equations",
                detection_difficulty="High - requires careful logical analysis",
                common_subjects=["logic", "set_theory", "algebra", "probability"]
            ),
            
            "domain_restriction_violation": ErrorType(
                name="domain_restriction_violation",
                description="Performing operations outside their valid domain",
                example="Dividing by expression that equals zero, or taking log of negative number",
                mathematical_context="Critical for functions, equations, and algebraic manipulations",
                detection_difficulty="Medium - requires domain awareness",
                common_subjects=["algebra", "analysis", "calculus", "functions"]
            ),
            
            "quantifier_confusion": ErrorType(
                name="quantifier_confusion",
                description="Mixing up universal and existential quantifiers",
                example="Proving existence when universality is needed, or vice versa",
                mathematical_context="Fundamental in formal proofs and mathematical logic",
                detection_difficulty="High - requires precise logical thinking",
                common_subjects=["logic", "analysis", "algebra", "set_theory"]
            ),
            
            "invalid_substitution": ErrorType(
                name="invalid_substitution",
                description="Substituting variables or expressions under invalid conditions",
                example="Substituting into inequality without considering sign changes",
                mathematical_context="Common in algebraic manipulations and equation solving",
                detection_difficulty="Medium - requires attention to substitution validity",
                common_subjects=["algebra", "calculus", "inequalities", "optimization"]
            ),
            
            "false_contrapositive": ErrorType(
                name="false_contrapositive",
                description="Incorrectly applying contrapositive or converse relationships", 
                example="Using 'not B implies not A' when only 'A implies B' is known",
                mathematical_context="Fundamental logical error in proof construction",
                detection_difficulty="High - requires understanding of logical equivalences",
                common_subjects=["logic", "proof_theory", "algebra", "geometry"]
            ),
            
            "invalid_inverse_operation": ErrorType(
                name="invalid_inverse_operation",
                description="Applying inverse operations without considering restrictions",
                example="Squaring both sides of equation without considering sign loss",
                mathematical_context="Common in equation solving and algebraic manipulations", 
                detection_difficulty="Medium - requires attention to operation properties",
                common_subjects=["algebra", "equations", "inequalities", "functions"]
            ),
            
            "composition_error": ErrorType(
                name="composition_error",
                description="Incorrectly composing functions or operations",
                example="Incorrectly computing f(g(x)) or applying chain rule incorrectly",
                mathematical_context="Critical for function operations and calculus",
                detection_difficulty="Medium - requires careful function analysis",
                common_subjects=["functions", "calculus", "analysis", "algebra"]
            ),
            
            "boundary_condition_error": ErrorType(
                name="boundary_condition_error", 
                description="Incorrectly handling boundary cases or edge conditions",
                example="Forgetting endpoint behavior in continuity or optimization problems",
                mathematical_context="Critical in analysis, optimization, and applied mathematics",
                detection_difficulty="High - boundary cases often overlooked",
                common_subjects=["calculus", "analysis", "optimization", "geometry"]
            )
        }
    
    def get_error_by_name(self, name: str) -> ErrorType:
        """Get error type by name."""
        return self.error_types.get(name)
    
    def get_errors_by_category(self, category: LogicalErrorCategory) -> List[ErrorType]:
        """Get all errors in a specific category."""
        category_mapping = {
            LogicalErrorCategory.GENERALIZATION: ["invalid_generalization"],
            LogicalErrorCategory.THEOREM_APPLICATION: ["theorem_misapplication"],
            LogicalErrorCategory.CASE_ANALYSIS: ["incomplete_case_analysis", "boundary_condition_error"],
            LogicalErrorCategory.LOGICAL_STRUCTURE: ["circular_reasoning", "false_contrapositive"],
            LogicalErrorCategory.DOMAIN_OPERATIONS: ["domain_restriction_violation", "invalid_inverse_operation"],
            LogicalErrorCategory.QUANTIFICATION: ["quantifier_confusion"],
            LogicalErrorCategory.SUBSTITUTION: ["invalid_substitution"],
            LogicalErrorCategory.EQUIVALENCE: ["false_equivalence", "composition_error"]
        }
        
        error_names = category_mapping.get(category, [])
        return [self.error_types[name] for name in error_names if name in self.error_types]
    
    def get_errors_by_subject(self, subject: str) -> List[ErrorType]:
        """Get all errors relevant to a mathematical subject."""
        return [
            error_type for error_type in self.error_types.values()
            if subject.lower() in [s.lower() for s in error_type.common_subjects]
        ]
    
    def get_all_error_names(self) -> List[str]:
        """Get list of all error type names."""
        return list(self.error_types.keys())
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about the error taxonomy."""
        total_errors = len(self.error_types)
        
        # Count by difficulty
        difficulty_counts = {}
        for error_type in self.error_types.values():
            difficulty = error_type.detection_difficulty.split(' - ')[0].lower()
            difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Count by subject
        subject_counts = {}
        for error_type in self.error_types.values():
            for subject in error_type.common_subjects:
                subject_counts[subject] = subject_counts.get(subject, 0) + 1
        
        return {
            "total_error_types": total_errors,
            "difficulty_distribution": difficulty_counts,
            "subject_distribution": subject_counts,
            "categories": len(LogicalErrorCategory)
        }
    
    def suggest_error_for_context(self, mathematical_context: str, 
                                 difficulty_preference: str = "high") -> List[str]:
        """Suggest appropriate error types for a given mathematical context."""
        context_keywords = mathematical_context.lower().split()
        
        relevant_errors = []
        for name, error_type in self.error_types.items():
            # Check if context matches error's mathematical context or subjects
            error_context = error_type.mathematical_context.lower()
            error_subjects = [s.lower() for s in error_type.common_subjects]
            
            context_match = any(keyword in error_context for keyword in context_keywords)
            subject_match = any(subject in context_keywords for subject in error_subjects)
            
            # Check difficulty preference
            difficulty_match = difficulty_preference.lower() in error_type.detection_difficulty.lower()
            
            if (context_match or subject_match) and difficulty_match:
                relevant_errors.append(name)
        
        return relevant_errors
    
    def get_prompt_examples(self, error_name: str) -> Dict[str, str]:
        """Get examples for prompt engineering for a specific error type."""
        error_type = self.get_error_by_name(error_name)
        if not error_type:
            return {}
        
        return {
            "error_type": error_name,
            "description": error_type.description,
            "example": error_type.example,
            "context_hint": error_type.mathematical_context,
            "detection_note": error_type.detection_difficulty,
            "subjects": ", ".join(error_type.common_subjects)
        }


# Create global instance for easy access
MATH_ERROR_TAXONOMY = MathematicalErrorTaxonomy()