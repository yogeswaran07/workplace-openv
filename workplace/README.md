---
title: Workplace Policy
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Workplace Policy Compliance

This OpenEnv environment evaluates whether an agent can act as a workplace policy
compliance analyst. The agent receives an employee request, a small set of company
policies, and must classify the request, approve or deny it, cite the relevant
policy IDs, and justify the decision.

The task models real operational work done in HR, finance, travel operations, and
employee support teams. It is intentionally not a game: the scoring checks whether
an agent can apply written policy to realistic requests with missing or sufficient
evidence.

## Action Space

`WorkplaceAction` is a typed Pydantic model:

```json
{
  "classification": "leave | travel | expense",
  "decision": "approved | denied",
  "rule_reference": "comma-separated policy IDs",
  "justification": "one sentence grounded in the policy facts"
}
```

## Observation Space

`WorkplaceObservation` is a typed Pydantic model containing:

- `task_id`: task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `objective`: concrete objective for the episode
- `request_text`: employee request to evaluate
- `available_policies`: policy snippets with stable IDs
- `attempts_remaining`: retry budget for the episode
- `previous_feedback`: deterministic grader feedback after a step
- `last_action_error`: missing-field error or `null`
- `reward`, `done`, and `metadata` from the OpenEnv observation base model

## Tasks

| Task | Difficulty | Objective |
| --- | --- | --- |
| `easy_leave_approval` | Easy | Approve or deny a routine annual leave request. |
| `medium_travel_approval` | Medium | Evaluate a domestic travel reimbursement against multiple limits. |
| `hard_expense_violation` | Hard | Detect a high-value expense claim with missing required evidence. |

Each task has a deterministic programmatic grader. Scores are always clamped to
the `[0.0, 1.0]` range and do not depend on randomness.

## Reward Function

The reward is dense and gives partial credit:

- Classification: `0.25`
- Approve or deny decision: `0.35`
- Cited policy evidence: `0.25`
- Justification grounded in task facts: `0.15`

Episodes end when the score reaches at least `0.95` or after 3 steps. This gives
agents useful feedback for retries while keeping episode boundaries short and
reproducible.

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r server/requirements.txt
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Check that it responds:

```bash
curl -s -X POST http://localhost:8000/reset -H 'Content-Type: application/json' -d '{}'
```

## Docker

Build and run:

```bash
docker build -t workplace-policy:latest .
docker run --rm -p 8000:8000 workplace-policy:latest
```

Validate the running server:

```bash
openenv validate http://localhost:8000
```

## OpenEnv Validation

From the repository root:

```bash
openenv validate
docker build -t workplace-policy:latest .
```

After deploying to a Hugging Face Space, run the submission validator with your
Space URL:

```bash
./scripts/validate-submission.sh https://your-space.hf.space .
```

## Baseline Inference

The baseline script is `inference.py` in the project root. It uses the
OpenAI-compatible client when `HF_TOKEN`, `OPENAI_API_KEY`, or `API_KEY` is set.
If no key is present, it uses a deterministic policy baseline so the script still
reproduces valid scores in local CI.

Environment variables:

- `API_BASE_URL`: OpenAI-compatible endpoint. Default: `https://router.huggingface.co/v1`
- `MODEL_NAME`: model identifier. Default: `Qwen/Qwen2.5-72B-Instruct`
- `HF_TOKEN`, `OPENAI_API_KEY`, or `API_KEY`: API credential
- `LOCAL_IMAGE_NAME`: optional local Docker image name for `from_docker_image()`

Run:

```bash
python inference.py
```

Stdout contains only these line types, in order for each task:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
[END] success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
```

Expected deterministic fallback scores:

| Task | Score |
| --- | --- |
| `easy_leave_approval` | `1.00` |
| `medium_travel_approval` | `1.00` |
| `hard_expense_violation` | `1.00` |

## Project Structure

```text
.
├── Dockerfile
├── README.md
├── __init__.py
├── client.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
└── server
    ├── __init__.py
    ├── app.py
    ├── requirements.txt
    └── workplace_environment.py
```
