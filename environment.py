"""
environment.py
==============
OpenEnv environment for code-review tasks.
"""

import random

from tasks import TASKS, TASK_INDEX
from grader import grade
from timepenalty import compute_time_penalty
from codeverifier import run_code

# Scores must be STRICTLY between 0 and 1 — never exactly 0.0 or 1.0
SCORE_FLOOR = 0.01
SCORE_CEIL  = 0.99

STEP_ALLOWANCES = {"easy": 2, "medium": 3, "hard": 4}


def _clamp(score: float) -> float:
    """Clamp reward to strictly (0, 1)."""
    return float(max(SCORE_FLOOR, min(SCORE_CEIL, float(score))))


def _default_axes() -> dict:
    """Return axis scores all set to SCORE_FLOOR."""
    return {
        "score":         SCORE_FLOOR,
        "issue_score":   SCORE_FLOOR,
        "line_score":    SCORE_FLOOR,
        "compile_score": SCORE_FLOOR,
        "test_score":    SCORE_FLOOR,
    }


class CodeReviewEnv:

    def __init__(self):
        self._task: dict | None        = None
        self._steps_taken              = 0
        self._done                     = False
        self._last_result: dict | None = None
        self._history: list            = []

    def reset(self, task_id: str | None = None) -> dict:
        if task_id is not None:
            if task_id not in TASK_INDEX:
                raise KeyError(
                    f"Unknown task_id '{task_id}'. "
                    f"Available ids: {list(TASK_INDEX.keys())}"
                )
            self._task = TASK_INDEX[task_id]
        else:
            self._task = random.choice(TASKS)

        self._steps_taken = 0
        self._done        = False
        self._last_result = None

        return self._build_observation()

    def step(self, action: dict) -> dict:
        if self._task is None:
            raise RuntimeError("No task loaded. Call reset() before step().")

        # Guard: episode already finished
        if self._done:
            last = self._last_result or {}
            reward = _clamp(last.get("final_reward", SCORE_FLOOR))
            return {
                "state":         self._build_observation(),
                "reward":        reward,
                "score":         reward,
                "done":          True,
                "issue_score":   _clamp(last.get("issue_score", SCORE_FLOOR)),
                "line_score":    _clamp(last.get("line_score", SCORE_FLOOR)),
                "compile_score": _clamp(last.get("compile_score", SCORE_FLOOR)),
                "test_score":    _clamp(last.get("test_score", SCORE_FLOOR)),
            }

        self._steps_taken += 1

        action_type = action.get("type", "submit") if isinstance(action, dict) else "submit"

        if action_type == "hint":
            return {
                "state":  self._build_observation(),
                "reward": SCORE_FLOOR,
                "done":   False,
                "hint":   (self._task or {}).get("bug_type", "unknown"),
                **_default_axes(),
            }

        if action_type == "run_test":
            test_cases = (self._task or {}).get("tests", [])
            if not test_cases:
                return {
                    "state":  self._build_observation(),
                    "reward": SCORE_FLOOR,
                    "done":   False,
                    **_default_axes(),
                }
            case   = test_cases[0]
            passed = run_code(action.get("fix", ""), case["input"])[0]
            return {
                "state":       self._build_observation(),
                "reward":      SCORE_FLOOR,
                "done":        False,
                "test_result": "pass" if passed else "fail",
                **_default_axes(),
            }

        # Submit — grade the action
        safe_action  = _validate_action(action)
        grade_result = grade(self._task, safe_action)
        base_reward  = grade_result.get("final_reward", SCORE_FLOOR)

        task = self._task or {}
        penalty_result = compute_time_penalty(
            base_reward=base_reward,
            steps_taken=self._steps_taken,
            difficulty=task.get("difficulty", "medium"),
        )

        # Clamp strictly between 0.01 and 0.99
        final_reward = _clamp(penalty_result.get("final_reward", SCORE_FLOOR))

        # Clamp individual axis scores
        clamped_issue   = _clamp(grade_result.get("issue_score", SCORE_FLOOR))
        clamped_line    = _clamp(grade_result.get("line_score", SCORE_FLOOR))
        clamped_compile = _clamp(grade_result.get("compile_score", SCORE_FLOOR))
        clamped_test    = _clamp(grade_result.get("test_score", SCORE_FLOOR))

        self._done = True
        self._last_result = {
            **grade_result,
            **penalty_result,
            "final_reward":  final_reward,
            "issue_score":   clamped_issue,
            "line_score":    clamped_line,
            "compile_score": clamped_compile,
            "test_score":    clamped_test,
            "steps_taken":   self._steps_taken,
        }

        self._history.append({
            "task_id":     task.get("id"),
            "reward":      final_reward,
            "penalty":     penalty_result.get("penalty", 0.0),
            "base_reward": base_reward,
        })

        return {
            "state":         self._build_observation(),
            "reward":        final_reward,
            "score":         final_reward,
            "done":          True,
            "issue_score":   clamped_issue,
            "line_score":    clamped_line,
            "compile_score": clamped_compile,
            "test_score":    clamped_test,
        }

    def state(self) -> dict:
        task          = self._task
        difficulty    = task.get("difficulty") if task else None
        steps_allowed = STEP_ALLOWANCES.get(difficulty, 0) if difficulty else 0

        # Filter result to only expose clamped numeric scores (no raw booleans)
        lr = self._last_result or {}
        safe_result = {}
        for k, v in lr.items():
            if k == "breakdown":
                continue
            if isinstance(v, float):
                safe_result[k] = _clamp(v)
            else:
                safe_result[k] = v

        return {
            "task_id":       task.get("id") if task else None,
            "difficulty":    difficulty,
            "steps_taken":   self._steps_taken,
            "steps_allowed": steps_allowed,
            "done":          self._done,
            "result":        safe_result,
        }

    def leaderboard(self) -> list[dict]:
        try:
            if not self._history:
                print("No runs yet")
                return []

            def _std(values: list[float]) -> float:
                n = len(values)
                if n < 2:
                    return 0.0
                mean     = sum(values) / n
                variance = sum((v - mean) ** 2 for v in values) / n
                return variance ** 0.5 if variance > 0 else 0.0

            def _f(v: float) -> float:
                return float(f"{v:.4f}")

            groups: dict[str, list] = {}
            for entry in self._history:
                tid = str(entry.get("task_id") or "unknown")
                groups.setdefault(tid, []).append(entry)

            rows = []
            for tid, entries in groups.items():
                rewards   = [float(e.get("reward", 0.0))  for e in entries] or [0.0]
                penalties = [float(e.get("penalty", 0.0)) for e in entries] or [0.0]
                n         = len(rewards)

                rows.append({
                    "task_id":           tid,
                    "mean_reward":       _f(sum(rewards) / n),
                    "min_reward":        _f(min(rewards)),
                    "max_reward":        _f(max(rewards)),
                    "avg_penalty":       _f(sum(penalties) / len(penalties)),
                    "consistency_score": _f(max(0.0, 1.0 - _std(rewards))),
                })

            rows.sort(key=lambda r: r["mean_reward"], reverse=True)

            header = (
                "TASK".ljust(26) +
                "AVG".ljust(8) +
                "MIN".ljust(8) +
                "MAX".ljust(8) +
                "PENALTY".ljust(12) +
                "CONSISTENCY".ljust(12)
            )
            print(header)
            print("-" * 74)
            for r in rows:
                print(
                    str(r["task_id"]).ljust(26) +
                    f"{float(r['mean_reward']):.2f}".ljust(8) +
                    f"{float(r['min_reward']):.2f}".ljust(8) +
                    f"{float(r['max_reward']):.2f}".ljust(8) +
                    f"{float(r['avg_penalty']):.2f}".ljust(12) +
                    f"{float(r['consistency_score']):.2f}".ljust(12)
                )

            return rows

        except Exception:
            print("No runs yet")
            return []

    def _build_observation(self) -> dict:
        task = self._task
        assert task is not None
        return {
            "task_id":     task.get("id"),
            "difficulty":  task.get("difficulty", "medium"),
            "title":       task.get("title", ""),
            "description": task.get("description", ""),
            "buggy_code":  task.get("code", ""),
        }


def _validate_action(action: dict) -> dict:
    if not isinstance(action, dict):
        action = {}

    action.setdefault("bug_line", -1)
    action.setdefault("issues", "")
    action.setdefault("fix", "")

    try:
        bug_line = int(action["bug_line"])
    except (TypeError, ValueError):
        bug_line = -1

    raw_issues = action.get("issues")
    if isinstance(raw_issues, list):
        issues = [str(i) for i in raw_issues if i is not None]
    elif isinstance(raw_issues, str) and raw_issues:
        issues = [raw_issues]
    else:
        issues = []

    fix = action["fix"] if isinstance(action["fix"], str) else ""

    return {
        "bug_line": bug_line,
        "issues":   issues,
        "fix":      fix,
    }