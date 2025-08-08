"""
Unified LateBench Data Schema
Defines the standardized format for all datasets in LateBench
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
import json
from datetime import datetime


@dataclass
class LateBenchStep:
    """Individual solution step in unified format"""
    step_number: int
    content: str
    importance: str = "medium"  # high, medium, low (from PRM800K)
    reasoning_type: str = "unknown"  # calculation, logic, algebraic, geometric, etc.
    is_modified: bool = False  # for error injection
    is_error: bool = False  # marks the error step
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchSource:
    """Source dataset information"""
    dataset: str  # numinamath, prm800k, etc.
    original_id: str  # original problem ID in source dataset
    difficulty: Union[int, float, str] = 3  # difficulty score, defaults to 3 (medium)
    subject: str = "mathematics"  # algebra, geometry, calculus, etc.
    competition: Optional[str] = None  # AMC, AIME, etc.
    year: Optional[int] = None  # competition year if available
    metadata: Dict[str, Any] = None  # source-specific metadata
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchProblem:
    """Problem statement with optional hint"""
    statement: str
    hint: Optional[str] = None  # Optional hint for solving the problem
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchSolution:
    """Complete solution with steps"""
    steps: List[LateBenchStep]
    final_answer: str
    total_steps: int = 0  # Calculated in __post_init__
    solution_method: str = "unknown"  # analytical, computational, etc.
    
    def __post_init__(self):
        self.total_steps = len(self.steps)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [step.to_dict() for step in self.steps],
            "final_answer": self.final_answer,
            "total_steps": self.total_steps,
            "solution_method": self.solution_method
        }


@dataclass
class LateBenchErrorInjection:
    """Simplified error injection data - single attempt only"""
    has_errors: bool = False
    injected_solution: Optional['LateBenchSolution'] = None  # Complete error-injected solution
    base_prompt: Optional[str] = None  # Base error injection prompt used
    manual_suggestion: Optional[str] = None  # Optional manual suggestion added to prompt
    target_error_step: Optional[int] = None  # Optional specific step number to target for error
    injection_timestamp: Optional[str] = None  # When injection was performed
    success: bool = False  # Whether injection succeeded
    error_info: Optional[Dict[str, Any]] = None  # Details like which step, explanation
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_errors": self.has_errors,
            "injected_solution": self.injected_solution.to_dict() if self.injected_solution else None,
            "base_prompt": self.base_prompt,
            "manual_suggestion": self.manual_suggestion,
            "target_error_step": self.target_error_step,
            "injection_timestamp": self.injection_timestamp,
            "success": self.success,
            "error_info": self.error_info
        }


@dataclass
class LateBenchCriticPrediction:
    """Critic model predictions for error detection"""
    has_errors: bool = False  # Critic's prediction of whether there are errors
    error_steps: List[int] = None  # Step numbers the critic identifies as errors
    confidence_scores: Optional[Dict[int, float]] = None  # Confidence per step (0.0-1.0)
    explanations: Optional[Dict[int, str]] = None  # Explanations for error steps
    processing_time: Optional[float] = None  # Time taken for prediction (seconds)
    model_version: Optional[str] = None  # Which critic model was used
    prediction_timestamp: Optional[str] = None  # When prediction was made (ISO format)
    success: bool = True  # Whether the prediction was successful
    
    def __post_init__(self):
        if self.error_steps is None:
            self.error_steps = []
        if self.confidence_scores is None:
            self.confidence_scores = {}
        if self.explanations is None:
            self.explanations = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchNaturalErrorAnnotation:
    """Human annotation of natural errors already present in the solution"""
    has_errors: bool = False  # Whether the solution contains natural errors
    first_error_step: Optional[int] = None  # The step number where the first error occurs
    error_description: Optional[str] = None  # Description of the first error (for future error injection guidance)
    propagated_error_steps: List[int] = field(default_factory=list)  # Step numbers where the original error propagated
    is_correct: bool = False  # Explicitly mark solution as correct (no natural errors)
    annotator: Optional[str] = None  # Who made the annotation
    timestamp: Optional[str] = None  # When annotation was made (ISO format)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchManualDecision:
    """Manual decision annotation by human evaluator"""
    decision: Optional[str] = None  # "yes", "maybe", "no"
    notes: Optional[str] = None  # Optional notes about the decision
    annotator: Optional[str] = None  # Who made the decision
    timestamp: Optional[str] = None  # When decision was made (ISO format)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchProcessing:
    """Processing and status information"""
    added_to_latebench: str  # ISO timestamp
    last_modified: str  # ISO timestamp
    status: str = "raw"  # raw, processed, annotated, finalized
    processor_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchExample:
    """Complete LateBench example in unified format"""
    id: str  # unique LateBench ID
    source: LateBenchSource
    problem: LateBenchProblem
    solution: LateBenchSolution
    error_injection: LateBenchErrorInjection
    processing: LateBenchProcessing
    original_error_steps: List[int] = field(default_factory=list)  # Ground truth errors in original solution (empty list = no errors)
    injected_error_steps: List[int] = field(default_factory=list)  # Ground truth errors from injection (empty list = no injection or no errors)
    critic_predictions_original: Optional[LateBenchCriticPrediction] = None  # Critic predictions for original solution
    critic_predictions_injected: Optional[LateBenchCriticPrediction] = None  # Critic predictions for error-injected solution
    manual_decision: Optional[LateBenchManualDecision] = None  # Manual human decision for the problem
    natural_error_annotation: Optional[LateBenchNaturalErrorAnnotation] = None  # Human annotation of natural errors in the solution
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "source": self.source.to_dict(),
            "problem": self.problem.to_dict(),
            "solution": self.solution.to_dict(),
            "error_injection": self.error_injection.to_dict(),
            "processing": self.processing.to_dict(),
            "original_error_steps": self.original_error_steps,
            "injected_error_steps": self.injected_error_steps,
            "critic_predictions_original": self.critic_predictions_original.to_dict() if self.critic_predictions_original else None,
            "critic_predictions_injected": self.critic_predictions_injected.to_dict() if self.critic_predictions_injected else None,
            "manual_decision": self.manual_decision.to_dict() if self.manual_decision else None,
            "natural_error_annotation": self.natural_error_annotation.to_dict() if self.natural_error_annotation else None
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LateBenchExample':
        """Create LateBenchExample from dictionary - simple deserialization with backward compatibility"""
        
        # Handle backward compatibility for old critic_predictions field
        critic_predictions_original = None
        critic_predictions_injected = None
        
        if data.get("critic_predictions_original"):
            critic_predictions_original = LateBenchCriticPrediction(**data["critic_predictions_original"])
        elif data.get("critic_predictions"):  # Backward compatibility
            critic_predictions_original = LateBenchCriticPrediction(**data["critic_predictions"])
            
        if data.get("critic_predictions_injected"):
            critic_predictions_injected = LateBenchCriticPrediction(**data["critic_predictions_injected"])
        
        # Handle manual decision
        manual_decision = None
        if data.get("manual_decision"):
            manual_decision = LateBenchManualDecision(**data["manual_decision"])
        
        # Handle natural error annotation
        natural_error_annotation = None
        if data.get("natural_error_annotation"):
            natural_error_annotation = LateBenchNaturalErrorAnnotation(**data["natural_error_annotation"])
        
        return cls(
            id=data["id"], 
            source=LateBenchSource(**data["source"]),
            problem=LateBenchProblem(
                statement=data["problem"]["statement"],
                hint=data["problem"].get("hint")
            ),
            solution=LateBenchSolution(
                steps=[LateBenchStep(**step) for step in data["solution"]["steps"]],
                final_answer=data["solution"]["final_answer"],
                total_steps=data["solution"]["total_steps"],
                solution_method=data["solution"]["solution_method"]
            ),
            error_injection=LateBenchErrorInjection(
                has_errors=data["error_injection"]["has_errors"],
                injected_solution=LateBenchSolution(
                    steps=[LateBenchStep(**step) for step in data["error_injection"]["injected_solution"]["steps"]],
                    final_answer=data["error_injection"]["injected_solution"]["final_answer"],
                    total_steps=data["error_injection"]["injected_solution"]["total_steps"],
                    solution_method=data["error_injection"]["injected_solution"]["solution_method"]
                ) if data["error_injection"].get("injected_solution") else None,
                base_prompt=data["error_injection"].get("base_prompt"),
                manual_suggestion=data["error_injection"].get("manual_suggestion"),
                target_error_step=data["error_injection"].get("target_error_step"),
                injection_timestamp=data["error_injection"].get("injection_timestamp"),
                success=data["error_injection"].get("success", False),
                error_info=data["error_injection"].get("error_info")
            ),
            processing=LateBenchProcessing(**data["processing"]),
            original_error_steps=data.get("original_error_steps", []),  # Backward compatibility with empty list default
            injected_error_steps=data.get("injected_error_steps", []),  # Backward compatibility with empty list default
            critic_predictions_original=critic_predictions_original,
            critic_predictions_injected=critic_predictions_injected,
            manual_decision=manual_decision,
            natural_error_annotation=natural_error_annotation
        )


def generate_latebench_id(source_dataset: str, original_id: str) -> str:
    """Generate unique LateBench ID"""
    # Create deterministic ID based on source
    import hashlib
    combined = f"{source_dataset}_{original_id}"
    hash_suffix = hashlib.md5(combined.encode()).hexdigest()[:8]
    return f"lb_{source_dataset}_{hash_suffix}"


def create_timestamp() -> str:
    """Create ISO format timestamp"""
    return datetime.utcnow().isoformat() + "Z"


# Error injection prompts are now handled by the ErrorInjector unified method


# Step importance color mapping for PRM800K
STEP_IMPORTANCE_COLORS = {
    "high": "#ff6b6b",      # red - critical steps
    "medium": "#4ecdc4",    # teal - important steps  
    "low": "#95e1d3",       # light green - supporting steps
    "unknown": "#6c757d"    # gray - unclassified
}

# Reasoning type classification
REASONING_TYPES = {
    "calculation": "Numerical computation",
    "algebraic": "Algebraic manipulation", 
    "geometric": "Geometric reasoning",
    "logical": "Logical deduction",
    "substitution": "Variable substitution",
    "simplification": "Expression simplification",
    "theorem_application": "Applying mathematical theorem",
    "case_analysis": "Case-by-case analysis",
    "unknown": "Unclassified reasoning"
}

