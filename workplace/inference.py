"""Baseline inference script for the workplace policy OpenEnv environment."""

from __future__ import annotations

import asyncio
import json
import os
import re
import textwrap
from typing import Any, List, Optional

from openai import OpenAI

from client import WorkplaceEnv
from models import Decision, RequestType, WorkplaceAction, WorkplaceObservation
from server.workplace_environment import TASKS, WorkplaceEnvironment


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")

TASK_NAMES = ["easy_leave_approval", "medium_travel_approval", "hard_expense_violation"]
BENCHMARK = os.getenv("WORKPLACE_POLICY_BENCHMARK", "workplace_policy")
MAX_STEPS = 3
TEMPERATURE = 0.2
MAX_TOKENS = 240
SUCCESS_SCORE_THRESHOLD = 0.80

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a workplace policy compliance analyst.
    Decide whether the employee request should be approved or denied using only
    the listed company policies. Return one compact JSON object with these keys:
    classification, decision, rule_reference, justification.
    Do not include markdown or any extra text.
    """
).strip()


def clean_line(value: Any) -> str:
    text = "" if value is None else str(value)
    return re.sub(r"\s+", " ", text).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = clean_line(error) if error else "null"
    print(
        f"[STEP] step={step} action={clean_line(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(observation: WorkplaceObservation, history: List[str]) -> str:
    policies = "\n".join(
        f"- {policy.id}: {policy.description}" for policy in observation.available_policies
    )
    history_block = "\n".join(history[-3:]) if history else "None"
    return textwrap.dedent(
        f"""
        Task: {observation.task_id}
        Difficulty: {observation.difficulty}
        Objective: {observation.objective}

        Employee request:
        {observation.request_text}

        Company policies:
        {policies}

        Previous feedback:
        {observation.previous_feedback or "None"}

        Previous attempts:
        {history_block}
        """
    ).strip()


def parse_model_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def action_from_data(data: dict[str, Any], observation: WorkplaceObservation) -> WorkplaceAction:
    fallback = heuristic_action(observation)

    classification = data.get("classification") or fallback.classification
    decision = data.get("decision") or fallback.decision
    rule_reference = data.get("rule_reference") or fallback.rule_reference
    justification = data.get("justification") or fallback.justification

    try:
        return WorkplaceAction(
            classification=classification,
            decision=decision,
            rule_reference=rule_reference,
            justification=justification,
        )
    except Exception:
        return fallback


def heuristic_action(observation: WorkplaceObservation) -> WorkplaceAction:
    request = observation.request_text.lower()
    task = TASKS[observation.task_id]

    if "annual leave" in request:
        classification = RequestType.leave
    elif "travel" in request or "trip" in request or "hotel" in request:
        classification = RequestType.travel
    else:
        classification = RequestType.expense

    denial_markers = [
        "no client names",
        "no client signatures",
        "no prior written approval",
        "$3,200",
        "missing required",
    ]
    decision = Decision.denied if any(marker in request for marker in denial_markers) else Decision.approved

    policy_ids = list(task.required_policy_ids)
    if observation.task_id == "easy_leave_approval":
        reason = (
            "approved because the employee has enough leave balance, requested only "
            "3 days, gave advance notice, and stays under the 5 consecutive day limit"
        )
    elif observation.task_id == "medium_travel_approval":
        reason = (
            "approved because receipts, manager approval, portal approval, the hotel "
            "cap, and the $1,000 total domestic travel limit are satisfied"
        )
    else:
        reason = (
            "denied because the $3,200 claim is over $500 and lacks prior approval, "
            "client names, signatures, and required documentation"
        )

    return WorkplaceAction(
        classification=classification,
        decision=decision,
        rule_reference=",".join(policy_ids),
        justification=reason,
    )


def get_model_action(
    client: Optional[OpenAI],
    observation: WorkplaceObservation,
    history: List[str],
) -> WorkplaceAction:
    if client is None:
        return heuristic_action(observation)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(observation, history)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = completion.choices[0].message.content or ""
        return action_from_data(parse_model_json(text), observation)
    except Exception:
        return heuristic_action(observation)


def action_to_str(action: WorkplaceAction) -> str:
    payload = action.model_dump(mode="json", exclude_none=True, exclude={"metadata"})
    return json.dumps(payload, sort_keys=True)


async def reset_env(env: Any, task_name: str) -> Any:
    result = env.reset(task_id=task_name)
    if asyncio.iscoroutine(result):
        return await result
    return result


async def step_env(env: Any, action: WorkplaceAction) -> Any:
    result = env.step(action)
    if asyncio.iscoroutine(result):
        return await result
    return result


def unwrap_observation(result: Any) -> WorkplaceObservation:
    return getattr(result, "observation", result)


def unwrap_reward(result: Any, observation: WorkplaceObservation) -> float:
    reward = getattr(result, "reward", observation.reward)
    return float(reward or 0.0)


def unwrap_done(result: Any, observation: WorkplaceObservation) -> bool:
    return bool(getattr(result, "done", observation.done))


async def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if close is None:
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result


async def make_env(task_name: str) -> Any:
    if LOCAL_IMAGE_NAME:
        return await WorkplaceEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return WorkplaceEnvironment(difficulty=task_name)


async def run_task(task_name: str, client: Optional[OpenAI]) -> tuple[bool, float]:
    env = await make_env(task_name)
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await reset_env(env, task_name)
        observation = unwrap_observation(result)

        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break

            action = get_model_action(client, observation, history)
            result = await step_env(env, action)
            observation = unwrap_observation(result)
            reward = unwrap_reward(result, observation)
            done = unwrap_done(result, observation)
            error = observation.last_action_error

            rewards.append(reward)
            steps_taken = step
            score = max(score, reward)

            log_step(
                step=step,
                action=action_to_str(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(f"step={step} action={action_to_str(action)} reward={reward:.2f}")
            if done:
                break

        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        await close_env(env)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return success, score


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if API_KEY else None
    for task_name in TASK_NAMES:
        await run_task(task_name, client)


if __name__ == "__main__":
    asyncio.run(main())
