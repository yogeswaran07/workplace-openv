"""Typed OpenEnv models for the workplace policy compliance environment."""

from enum import Enum
from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field, field_validator


class Decision(str, Enum):
    """Supported policy decisions."""

    approved = "approved"
    denied = "denied"


class RequestType(str, Enum):
    """Supported workplace request categories."""

    leave = "leave"
    travel = "travel"
    expense = "expense"


class PolicyEvidence(BaseModel):
    """A policy snippet visible to the agent."""

    id: str = Field(..., description="Stable policy identifier, such as L1 or E3")
    description: str = Field(..., description="Human-readable policy text")


class WorkplaceAction(Action):
    """Agent action: classify the request, make a decision, and cite evidence."""

    classification: Optional[RequestType] = Field(
        default=None,
        description="Request category: leave, travel, or expense",
    )
    decision: Optional[Decision] = Field(
        default=None,
        description="Final decision: approved or denied",
    )
    rule_reference: Optional[str] = Field(
        default=None,
        description="Comma-separated policy IDs cited for the decision, such as L1,L2",
    )
    justification: Optional[str] = Field(
        default=None,
        description="One-sentence explanation grounded in the cited policies",
    )

    @field_validator("rule_reference")
    @classmethod
    def normalize_rule_reference(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None


class WorkplaceReward(BaseModel):
    """Deterministic grader output in the required 0.0 to 1.0 range."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    classification_score: float = Field(default=0.0, ge=0.0, le=0.25)
    decision_score: float = Field(default=0.0, ge=0.0, le=0.35)
    evidence_score: float = Field(default=0.0, ge=0.0, le=0.25)
    justification_score: float = Field(default=0.0, ge=0.0, le=0.15)
    feedback: str = Field(default="", description="Deterministic feedback for retrying")


class WorkplaceObservation(Observation):
    """Observation returned by reset() and step()."""

    task_id: str = Field(default="", description="Current task identifier")
    difficulty: str = Field(default="", description="Difficulty label")
    objective: str = Field(default="", description="Concrete task objective")
    request_text: str = Field(default="", description="Employee request to evaluate")
    available_policies: List[PolicyEvidence] = Field(
        default_factory=list,
        description="Policy snippets the agent may cite",
    )
    attempts_remaining: int = Field(default=0, ge=0)
    previous_feedback: str = Field(default="")
    last_action_error: Optional[str] = Field(default=None)
    expected_format: str = Field(
        default=(
            "Return JSON with classification, decision, rule_reference, and justification."
        )
    )
