"""
Unified LateBench Data Schema
Defines the standardized format for all datasets in LateBench
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
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
    difficulty: Union[int, float, str]  # normalized or original difficulty
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
    """Problem statement and context"""
    statement: str
    constraints: Optional[str] = None  # additional constraints
    context: Optional[str] = None  # problem context/background
    figures: Optional[List[str]] = None  # figure descriptions/paths
    
    def __post_init__(self):
        if self.figures is None:
            self.figures = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchSolution:
    """Complete solution with steps"""
    steps: List[LateBenchStep]
    final_answer: str
    total_steps: int
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
class ErrorInjectionAttempt:
    """Single manual error injection attempt"""
    attempt_number: int
    user_suggestion: str
    user_remarks: str
    injection_result: Dict[str, Any]  # result from error injector
    timestamp: str
    success: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LateBenchErrorInjection:
    """Error injection data and manual annotations"""
    has_errors: bool = False
    error_info: Optional[Dict[str, Any]] = None  # populated when errors injected
    manual_attempts: List[ErrorInjectionAttempt] = None
    final_decision: Optional[str] = None  # yes, maybe, no
    decision_timestamp: Optional[str] = None
    custom_suggestions: List[str] = None  # saved error suggestions
    
    def __post_init__(self):
        if self.manual_attempts is None:
            self.manual_attempts = []
        if self.custom_suggestions is None:
            self.custom_suggestions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_errors": self.has_errors,
            "error_info": self.error_info,
            "manual_attempts": [attempt.to_dict() for attempt in self.manual_attempts],
            "final_decision": self.final_decision,
            "decision_timestamp": self.decision_timestamp,
            "custom_suggestions": self.custom_suggestions
        }


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "source": self.source.to_dict(),
            "problem": self.problem.to_dict(),
            "solution": self.solution.to_dict(),
            "error_injection": self.error_injection.to_dict(),
            "processing": self.processing.to_dict()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LateBenchExample':
        """Create LateBenchExample from dictionary"""
        # Reconstruct steps
        steps = [LateBenchStep(**step) for step in data["solution"]["steps"]]
        
        # Reconstruct manual attempts
        attempts = [ErrorInjectionAttempt(**attempt) for attempt in data["error_injection"]["manual_attempts"]]
        
        return cls(
            id=data["id"],
            source=LateBenchSource(**data["source"]),
            problem=LateBenchProblem(**data["problem"]),
            solution=LateBenchSolution(
                steps=steps,
                final_answer=data["solution"]["final_answer"],
                total_steps=data["solution"]["total_steps"],
                solution_method=data["solution"].get("solution_method", "unknown")
            ),
            error_injection=LateBenchErrorInjection(
                has_errors=data["error_injection"]["has_errors"],
                error_info=data["error_injection"]["error_info"],
                manual_attempts=attempts,
                final_decision=data["error_injection"]["final_decision"],
                decision_timestamp=data["error_injection"]["decision_timestamp"],
                custom_suggestions=data["error_injection"]["custom_suggestions"]
            ),
            processing=LateBenchProcessing(**data["processing"])
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

# Difficulty normalization
def normalize_difficulty(difficulty: Union[int, float, str], source_dataset: str) -> float:
    """Normalize difficulty to 0-5 scale"""
    if source_dataset == "numinamath":
        # NuminaMath doesn't have explicit difficulty, estimate from complexity
        return 3.0  # default medium difficulty
    elif source_dataset == "prm800k":
        # PRM800K uses 1-5 scale, return as is
        try:
            return float(difficulty)
        except:
            return 3.0
    else:
        return 3.0  # default