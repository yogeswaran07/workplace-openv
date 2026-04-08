"""Typed async client for the workplace policy OpenEnv server."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import WorkplaceAction, WorkplaceObservation
except ImportError:  # pragma: no cover - direct repo execution path
    from models import WorkplaceAction, WorkplaceObservation


class WorkplaceEnv(EnvClient[WorkplaceAction, WorkplaceObservation, State]):
    """WebSocket client for a running workplace policy environment."""

    def _step_payload(self, action: WorkplaceAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[WorkplaceObservation]:
        obs_data = payload.get("observation", {})
        observation = WorkplaceObservation(
            **obs_data,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(**payload)


WorkplacePolicyEnv = WorkplaceEnv
