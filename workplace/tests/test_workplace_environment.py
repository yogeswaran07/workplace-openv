from workplace_policy_env import Decision, RequestType, WorkplaceAction, WorkplaceEnvironment
from workplace_policy_env.server.workplace_environment import TASKS


def test_all_tasks_can_score_full_credit() -> None:
    for task_id, task in TASKS.items():
        env = WorkplaceEnvironment(difficulty=task_id)
        observation = env.reset(task_id=task_id)

        result = env.step(
            WorkplaceAction(
                classification=task.correct_classification,
                decision=task.correct_decision,
                rule_reference=",".join(task.required_policy_ids),
                justification=(
                    f"{task.correct_decision.value} because the policy facts include "
                    f"{', '.join(task.justification_terms)}."
                ),
            )
        )

        assert observation.task_id == task_id
        assert result.reward is not None
        assert float(result.reward) >= 0.95
        assert result.done is True


def test_partial_credit_for_correct_classification_only() -> None:
    env = WorkplaceEnvironment(difficulty="hard")
    env.reset()

    result = env.step(
        WorkplaceAction(
            classification=RequestType.expense,
            decision=Decision.approved,
            rule_reference="L1",
            justification="bad",
        )
    )

    assert result.reward == 0.25
    assert result.done is False
    assert "decision" in result.previous_feedback
