"""Workplace policy compliance OpenEnv environment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State

try:
    from ..models import (
        Decision,
        PolicyEvidence,
        RequestType,
        WorkplaceAction,
        WorkplaceObservation,
        WorkplaceReward,
    )
except ImportError:  # pragma: no cover - used when running as a loose repo
    from models import (
        Decision,
        PolicyEvidence,
        RequestType,
        WorkplaceAction,
        WorkplaceObservation,
        WorkplaceReward,
    )


@dataclass(frozen=True)
class PolicyTask:
    task_id: str
    difficulty: str
    objective: str
    request_text: str
    policies: tuple[PolicyEvidence, ...]
    correct_classification: RequestType
    correct_decision: Decision
    required_policy_ids: tuple[str, ...]
    justification_terms: tuple[str, ...]
    max_steps: int = 3


TASKS: dict[str, PolicyTask] = {
    "easy_leave_approval": PolicyTask(
        task_id="easy_leave_approval",
        difficulty="easy",
        objective="Approve or deny a routine annual leave request.",
        request_text=(
            "Maya Rao requests 3 consecutive days of annual leave next month for a "
            "family event. She has 8 unused leave days and submitted the request "
            "10 days before the planned absence."
        ),
        policies=(
            PolicyEvidence(
                id="L1",
                description=(
                    "Annual leave up to 5 consecutive working days is approved when "
                    "the employee has enough leave balance."
                ),
            ),
            PolicyEvidence(
                id="L2",
                description="Annual leave must be submitted at least 3 days in advance.",
            ),
            PolicyEvidence(
                id="E1",
                description="Expenses over $500 require written approval before purchase.",
            ),
        ),
        correct_classification=RequestType.leave,
        correct_decision=Decision.approved,
        required_policy_ids=("L1", "L2"),
        justification_terms=("balance", "advance", "3 days", "5 consecutive"),
    ),
    "medium_travel_approval": PolicyTask(
        task_id="medium_travel_approval",
        difficulty="medium",
        objective="Evaluate a domestic travel reimbursement against multiple limits.",
        request_text=(
            "Jordan Lee requests reimbursement for a domestic client trip: $310 train "
            "fare, $220 hotel for one night, and $180 meals. Receipts are attached, "
            "and the trip was pre-approved by Jordan's manager in the travel portal."
        ),
        policies=(
            PolicyEvidence(
                id="T1",
                description=(
                    "Domestic business travel up to $1,000 total is approved when "
                    "receipts and manager approval are present."
                ),
            ),
            PolicyEvidence(
                id="T2",
                description="Hotel reimbursement is capped at $250 per night.",
            ),
            PolicyEvidence(
                id="T3",
                description="Travel must be booked or approved through the travel portal.",
            ),
        ),
        correct_classification=RequestType.travel,
        correct_decision=Decision.approved,
        required_policy_ids=("T1", "T2", "T3"),
        justification_terms=("receipts", "manager", "$1,000", "hotel", "portal"),
    ),
    "hard_expense_violation": PolicyTask(
        task_id="hard_expense_violation",
        difficulty="hard",
        objective="Detect a high-value expense claim with missing required evidence.",
        request_text=(
            "Alex Chen submitted a $3,200 client entertainment dinner claim from last "
            "Thursday. The claim has no client names, no client signatures, and no "
            "prior written approval attached."
        ),
        policies=(
            PolicyEvidence(
                id="E1",
                description=(
                    "Expenses over $500 require written approval before they are incurred."
                ),
            ),
            PolicyEvidence(
                id="E2",
                description=(
                    "Client entertainment claims must include client names and signatures."
                ),
            ),
            PolicyEvidence(
                id="E3",
                description="Claims missing required documentation must be denied.",
            ),
        ),
        correct_classification=RequestType.expense,
        correct_decision=Decision.denied,
        required_policy_ids=("E1", "E2", "E3"),
        justification_terms=("over $500", "prior", "client names", "signatures", "documentation"),
    ),
}

TASK_ALIASES = {
    "easy": "easy_leave_approval",
    "leave": "easy_leave_approval",
    "medium": "medium_travel_approval",
    "travel": "medium_travel_approval",
    "hard": "hard_expense_violation",
    "expense": "hard_expense_violation",
}


def resolve_task(task_name: Optional[str]) -> PolicyTask:
    """Resolve a task id, difficulty, or alias to a task definition."""

    key = (task_name or "easy_leave_approval").strip().lower()
    key = TASK_ALIASES.get(key, key)
    if key not in TASKS:
        valid = ", ".join(sorted([*TASKS.keys(), *TASK_ALIASES.keys()]))
        raise ValueError(f"Unknown task '{task_name}'. Valid tasks: {valid}")
    return TASKS[key]


class WorkplaceEnvironment(Environment[WorkplaceAction, WorkplaceObservation, State]):
    """Environment for evaluating HR and finance policy decision agents."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, difficulty: str = "easy"):
        super().__init__()
        self._default_task_name = difficulty
        self._task = resolve_task(difficulty)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._last_reward = WorkplaceReward()
        self._last_feedback = ""
        self._last_action_error: Optional[str] = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **_: object,
    ) -> WorkplaceObservation:
        """Reset the episode and return a clean task observation."""

        del seed
        self._task = resolve_task(task_id or difficulty or self._default_task_name)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            done=False,
            last_score=0.0,
        )
        self._done = False
        self._last_reward = WorkplaceReward()
        self._last_feedback = ""
        self._last_action_error = None
        return self._observation(reward=0.0, done=False)

    def step(
        self,
        action: WorkplaceAction,
        timeout_s: Optional[float] = None,
        **_: object,
    ) -> WorkplaceObservation:
        """Grade one policy decision action and return partial-credit feedback."""

        del timeout_s
        if self._done:
            self._last_action_error = "Episode already done. Call reset() before step()."
            return self._observation(reward=0.0, done=True)

        self._state.step_count += 1
        reward = self._grade(action)
        self._last_reward = reward
        self._last_feedback = reward.feedback
        self._last_action_error = self._missing_action_error(action)
        self._done = reward.score >= 0.95 or self._state.step_count >= self._task.max_steps
        self._state = State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            done=self._done,
            last_score=reward.score,
        )
        return self._observation(reward=reward.score, done=self._done)

    @property
    def state(self) -> State:
        """Return current episode state."""

        return self._state

    def get_metadata(self) -> EnvironmentMetadata:
        """Return metadata exposed by the OpenEnv server."""

        return EnvironmentMetadata(
            name="workplace_policy",
            description=(
                "Classify workplace requests, make approve/deny decisions, and cite "
                "deterministic policy evidence across easy, medium, and hard tasks."
            ),
            version="1.0.0",
            author="OpenEnv hackathon submission",
        )

    def _observation(self, reward: float, done: bool) -> WorkplaceObservation:
        attempts_remaining = max(self._task.max_steps - self._state.step_count, 0)
        return WorkplaceObservation(
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            objective=self._task.objective,
            request_text=self._task.request_text,
            available_policies=list(self._task.policies),
            attempts_remaining=attempts_remaining,
            previous_feedback=self._last_feedback,
            last_action_error=self._last_action_error,
            done=done,
            reward=round(float(reward), 4),
            metadata={
                "score_components": self._last_reward.model_dump(),
                "correct_classification": self._task.correct_classification.value,
                "correct_decision": self._task.correct_decision.value,
                "required_policy_ids": list(self._task.required_policy_ids),
            },
        )

    def _grade(self, action: WorkplaceAction) -> WorkplaceReward:
        classification_score = (
            0.25 if action.classification == self._task.correct_classification else 0.0
        )
        decision_score = 0.35 if action.decision == self._task.correct_decision else 0.0

        cited_ids = self._extract_policy_ids(action)
        required_ids = set(self._task.required_policy_ids)
        evidence_hits = required_ids.intersection(cited_ids)
        evidence_score = 0.25 * (len(evidence_hits) / len(required_ids))

        justification_score = self._score_justification(action.justification or "")
        score = classification_score + decision_score + evidence_score + justification_score
        score = round(min(max(score, 0.0), 1.0), 4)

        feedback = self._feedback(
            action=action,
            cited_ids=cited_ids,
            evidence_hits=evidence_hits,
            justification_score=justification_score,
        )
        return WorkplaceReward(
            score=score,
            classification_score=round(classification_score, 4),
            decision_score=round(decision_score, 4),
            evidence_score=round(evidence_score, 4),
            justification_score=round(justification_score, 4),
            feedback=feedback,
        )

    def _score_justification(self, justification: str) -> float:
        text = justification.lower()
        if len(text.strip()) < 20:
            return 0.0

        term_hits = sum(1 for term in self._task.justification_terms if term.lower() in text)
        term_score = 0.10 * min(term_hits / max(len(self._task.justification_terms), 1), 1.0)
        decision_word = self._task.correct_decision.value
        decision_score = 0.05 if decision_word in text else 0.0
        return round(term_score + decision_score, 4)

    @staticmethod
    def _extract_policy_ids(action: WorkplaceAction) -> set[str]:
        if not action.rule_reference:
            return set()
        return set(re.findall(r"[A-Z]\d+", action.rule_reference.upper()))

    def _missing_action_error(self, action: WorkplaceAction) -> Optional[str]:
        missing = []
        if action.classification is None:
            missing.append("classification")
        if action.decision is None:
            missing.append("decision")
        if not action.rule_reference:
            missing.append("policy evidence")
        if not (action.justification or "").strip():
            missing.append("justification")
        if missing:
            return "Missing required action field(s): " + ", ".join(missing)
        return None

    def _feedback(
        self,
        action: WorkplaceAction,
        cited_ids: Iterable[str],
        evidence_hits: set[str],
        justification_score: float,
    ) -> str:
        messages: list[str] = []
        if action.classification != self._task.correct_classification:
            messages.append("classification does not match the request type")
        if action.decision != self._task.correct_decision:
            messages.append("decision does not match the policy outcome")

        missing_evidence = set(self._task.required_policy_ids) - set(evidence_hits)
        if missing_evidence:
            messages.append("missing policy evidence: " + ",".join(sorted(missing_evidence)))
        if set(cited_ids) and not set(cited_ids).intersection(self._task.required_policy_ids):
            messages.append("cited policies are not relevant to this task")
        if justification_score < 0.10:
            messages.append("justification should cite the concrete policy facts")
        if not messages:
            return "all grading criteria satisfied"
        return "; ".join(messages)


WorkplacePolicyEnvironment = WorkplaceEnvironment
