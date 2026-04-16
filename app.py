"""
app.py — Smart Code Review · Hugging Face Space
================================================
Exposes:
  - FastAPI REST endpoints  /reset  /step  /state  (for OpenEnv automated checks)
  - Gradio web UI at /ui   (for human interaction)
"""

import json
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import gradio as gr
from environment import CodeReviewEnv
from tasks import TASKS

# ══════════════════════════════════════════════════════════════════════
# FastAPI app — OpenEnv REST interface
# ══════════════════════════════════════════════════════════════════════

api = FastAPI(title="Smart Code Review — OpenEnv API")

_api_env = CodeReviewEnv()

SCORE_FLOOR = 0.01
SCORE_CEIL  = 0.99


def _clamp(score) -> float:
    try:
        return float(max(SCORE_FLOOR, min(SCORE_CEIL, float(score))))
    except Exception:
        return SCORE_FLOOR


def _clamp_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return result
    if "reward" in result:
        result["reward"] = _clamp(result["reward"])
    return result


@api.post("/reset")
async def api_reset(request: Request):
    global _api_env
    try:
        body = await request.json()
        task_id = body.get("task_id", None) if isinstance(body, dict) else None
    except Exception:
        task_id = None
    try:
        _api_env = CodeReviewEnv()
        obs = _api_env.reset(task_id)
        return JSONResponse(content=obs)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@api.post("/step")
async def api_step(request: Request):
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    global _api_env
    if _api_env._task is None:
        _api_env = CodeReviewEnv()
        _api_env.reset()

    action = {
        "bug_line": body.get("bug_line", -1),
        "issues":   body.get("issues",   []),
        "fix":      body.get("fix",       ""),
    }

    try:
        result = _api_env.step(action)
        result = _clamp_result(result)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@api.post("/review")
async def api_review(request: Request):
    """
    Accept any Python code, generate a task dynamically,
    run the AI agent, grade it, and return full results.
    """
    from inference import get_action
    from codeverifier import run_dynamic_test

    # 1. Parse body
    try:
        body = await request.json()
        code = body.get("code", "").strip() if isinstance(body, dict) else ""
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid request body"})

    if not code:
        return JSONResponse(status_code=400, content={"error": "No code provided"})

    # 2. Load dynamic task into env (validates syntax + generates task via LLM)
    try:
        env = CodeReviewEnv()
        obs = env.reset_from_code(code)
    except ValueError as e:
        # Syntax error from validate_syntax()
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Task generation failed: {e}"})

    # 3. Run AI agent on the buggy code
    try:
        action = get_action(obs["buggy_code"])
        action.pop("used_reflection", None)
        action.pop("_initial_action", None)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Agent failed: {e}"})

    # 4. Grade via existing step()
    try:
        result = env.step(action)
        result = _clamp_result(result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Grading failed: {e}"})

    # 5. Run dynamic test cases for UI display
    task         = env._task or {}
    test_cases   = task.get("test_cases", [])
    test_results = []
    for tc in test_cases:
        try:
            tr = run_dynamic_test(action.get("fix", ""), tc)
            test_results.append({
                "input":    str(tc.get("input", "")),
                "expected": str(tc.get("expected", "")),
                "got":      str(tr.get("result", "error")),
                "passed":   bool(tr.get("passed", False)),
                "error":    tr.get("error"),
            })
        except Exception as e:
            test_results.append({
                "input":    str(tc.get("input", "")),
                "expected": str(tc.get("expected", "")),
                "got":      "error",
                "passed":   False,
                "error":    str(e),
            })

    # 6. Return full response
    lr = env._last_result or {}
    return JSONResponse(content={
        "bug_type":    task.get("bug_type", "unknown"),
        "bug_line":    task.get("bug_line", 0),
        "description": task.get("description", ""),
        "fix":         action.get("fix", ""),
        "issues":      action.get("issues", []),
        "score":       result.get("reward", SCORE_FLOOR),
        "issue_score":    _clamp(lr.get("issue_score",   SCORE_FLOOR)),
        "line_score":     _clamp(lr.get("line_score",    SCORE_FLOOR)),
        "compile_score":  _clamp(lr.get("compile_score", SCORE_FLOOR)),
        "test_score":     _clamp(lr.get("test_score",    SCORE_FLOOR)),
        "tests_passed":   lr.get("tests_passed", 0),
        "tests_total":    lr.get("tests_total",  0),
        "test_results":   test_results,
    })


@api.get("/state")
def api_state():
    try:
        return JSONResponse(content=_api_env.state())
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


# ══════════════════════════════════════════════════════════════════════
# Gradio UI helpers
# ══════════════════════════════════════════════════════════════════════

TASK_IDS   = [t["id"] for t in TASKS]
TASK_LABEL = {t["id"]: f"[{t['difficulty'].upper()}] {t['title']}" for t in TASKS}


def load_task(task_id: str, state: dict) -> tuple:
    env = CodeReviewEnv()
    obs = env.reset(task_id)
    state["env"] = env
    state["obs"] = obs
    diff_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(obs["difficulty"], "⚪")
    header = f"{diff_emoji} **{obs['title']}**  ·  `{obs['task_id']}`"
    return header, obs["description"], obs["buggy_code"], "", "", "", "", "", state


def review_custom_code(code: str, state: dict) -> tuple:
    """Gradio handler for arbitrary code review."""
    if not code.strip():
        return "⚠️ Paste some Python code first.", "", "", "", state

    from inference import get_action
    from codeverifier import run_dynamic_test

    try:
        env = CodeReviewEnv()
        obs = env.reset_from_code(code)
    except ValueError as e:
        return f"⚠️ {e}", "", "", "", state
    except Exception as e:
        return f"⚠️ Task generation failed: {e}", "", "", "", state

    try:
        action = get_action(obs["buggy_code"])
        action.pop("used_reflection", None)
        action.pop("_initial_action", None)
    except Exception as e:
        return f"⚠️ Agent failed: {e}", "", "", "", state

    try:
        result = env.step(action)
    except Exception as e:
        return f"⚠️ Grading failed: {e}", "", "", "", state

    lr = env._last_result or {}
    state["env"]         = env
    state["last_result"] = result

    # Build test results markdown
    task       = env._task or {}
    test_cases = task.get("test_cases", [])
    test_md    = ""
    if test_cases:
        test_md = "\n\n**Test results:**\n| Input | Expected | Got | Pass |\n|---|---|---|---|\n"
        for tc in test_cases:
            try:
                tr     = run_dynamic_test(action.get("fix", ""), tc)
                got    = str(tr.get("result", "error"))
                passed = "✅" if tr.get("passed") else "❌"
            except Exception:
                got    = "error"
                passed = "❌"
            test_md += f"| `{tc.get('input','')}` | `{tc.get('expected','')}` | `{got}` | {passed} |\n"

    bug_info = (
        f"**Bug type:** `{task.get('bug_type', 'unknown')}`  ·  "
        f"**Bug line:** `{task.get('bug_line', '?')}`\n\n"
        f"**Description:** {task.get('description', '')}"
        f"{test_md}"
    )

    return (
        _reward_bar(result["reward"]),
        _scores_table(lr),
        bug_info,
        action.get("fix", ""),
        state,
    )


def run_inference(task_id: str, state: dict) -> tuple:
    env = state.get("env")
    obs = state.get("obs", {})
    if env is None:
        return "⚠️ Load a task first.", "", "", "", state
    buggy_code = obs.get("buggy_code", "")
    from inference import get_action
    action = get_action(buggy_code)
    action.pop("used_reflection", None)
    action.pop("_initial_action", None)
    result = env.step(action)
    lr     = env._last_result or {}
    state["last_result"] = result
    return (
        _reward_bar(result["reward"]),
        _scores_table(lr),
        f"```json\n{json.dumps(action, indent=2)}\n```",
        action.get("fix", ""),
        state,
    )


def run_manual(bug_line: str, issues: str, fix: str, state: dict) -> tuple:
    env = state.get("env")
    if env is None:
        return "⚠️ Load a task first.", "", state
    try:
        bl = int(bug_line)
    except (ValueError, TypeError):
        bl = -1
    issues_list = [i.strip() for i in issues.split(",") if i.strip()]
    action = {"bug_line": bl, "issues": issues_list, "fix": fix}
    result = env.step(action)
    lr     = env._last_result or {}
    state["last_result"] = result
    return _reward_bar(result["reward"]), _scores_table(lr), state


def _reward_bar(reward: float) -> str:
    pct = int(reward * 100)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    return f"### Reward: **{reward:.2f}** / 1.00\n\n`{bar}` {pct}%"


def _scores_table(lr: dict) -> str:
    if not lr:
        return ""
    rows = [
        ("Issue description",   lr.get("issue_score",   0), 30),
        ("Line identification", lr.get("line_score",    0), 20),
        ("Compile check",       lr.get("compile_score", 0), 20),
        ("Test cases",          lr.get("test_score",    0), 30),
    ]
    tp = lr.get("tests_passed", 0)
    tt = lr.get("tests_total",  0)
    md = "| Axis | Score | Weight |\n|---|---|---|\n"
    for label, score, weight in rows:
        emoji = "✅" if score >= 0.99 else "🟡" if score > 0 else "❌"
        extra = f" ({tp}/{tt} tests)" if label == "Test cases" else ""
        md += f"| {emoji} {label}{extra} | `{score:.2f}` | {weight}% |\n"
    penalty = lr.get("penalty",       0)
    steps   = lr.get("steps_taken",   0)
    allowed = lr.get("steps_allowed", 0)
    md += f"\n**Steps:** {steps} / {allowed} allowed"
    if penalty:
        md += f"  ·  **Penalty:** −{penalty:.2f}"
    return md


# ══════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════

with gr.Blocks() as gradio_app:

    state = gr.State({})

    gr.Markdown("# 🧠 Smart Code Review")
    gr.Markdown(
        "Paste any Python function — the AI finds the bug, "
        "generates a fix, runs it, and scores the result."
    )

    with gr.Tabs():

        with gr.TabItem("🔍 Review Any Code"):
            gr.Markdown("> Paste any Python function. Works on code you write, not just preloaded tasks.")
            custom_code_input = gr.Code(
                label="Paste your Python code here",
                language="python",
                lines=14,
                value="def sum_list(numbers):\n    total = 0\n    for i in range(1, len(numbers)):\n        total += numbers[i]\n    return total",
            )
            custom_btn    = gr.Button("Review This Code ⚡", variant="primary")
            custom_reward = gr.Markdown("*Paste code and click Review.*")
            custom_scores = gr.Markdown()
            custom_bug    = gr.Markdown()
            with gr.Accordion("Fixed Code", open=False):
                custom_fix = gr.Code(language="python", lines=14, interactive=False)

        with gr.TabItem("📋 Preloaded Tasks"):
            gr.Markdown("> Choose a preloaded buggy task and run the AI agent or submit your own fix.")
            with gr.Row():
                task_dropdown = gr.Dropdown(
                    choices=[(TASK_LABEL[tid], tid) for tid in TASK_IDS],
                    value=TASK_IDS[0], label="Select Task", scale=4,
                )
                load_btn = gr.Button("Load Task ▶️", variant="primary", scale=1)

            task_header = gr.Markdown("*Load a task to begin.*")

            with gr.Row():
                with gr.Column(scale=1):
                    task_desc  = gr.Textbox(label="Task Description", lines=3, interactive=False)
                with gr.Column(scale=2):
                    buggy_code = gr.Code(label="Buggy Code", language="python",
                                         lines=14, interactive=False)

            gr.Markdown("---")

            with gr.Tabs():
                with gr.TabItem("🤖 AI Agent"):
                    agent_btn    = gr.Button("Run AI Agent ⚡", variant="primary")
                    agent_reward = gr.Markdown("*Run the agent to see results.*")
                    agent_scores = gr.Markdown()
                    with gr.Accordion("Agent Action (JSON)", open=False):
                        agent_action_md = gr.Markdown()
                    with gr.Accordion("Agent's Fixed Code", open=False):
                        agent_fix = gr.Code(language="python", lines=16, interactive=False)

                with gr.TabItem("✍️ Manual Submission"):
                    with gr.Row():
                        manual_line   = gr.Textbox(label="Bug Line Number",
                                                   placeholder="e.g. 3", scale=1)
                        manual_issues = gr.Textbox(label="Issues (comma-separated)",
                                                   placeholder="e.g. off-by-one", scale=3)
                    manual_fix    = gr.Code(label="Your Fixed Code", language="python",
                                            lines=14, value="# Paste your corrected code here")
                    manual_btn    = gr.Button("Submit Fix ✅", variant="primary")
                    manual_reward = gr.Markdown()
                    manual_scores = gr.Markdown()

    # Wire up custom review tab
    custom_btn.click(
        review_custom_code,
        inputs=[custom_code_input, state],
        outputs=[custom_reward, custom_scores, custom_bug, custom_fix, state],
    )

    # Wire up preloaded tasks tab
    load_btn.click(
        load_task,
        inputs=[task_dropdown, state],
        outputs=[task_header, task_desc, buggy_code,
                 agent_reward, agent_scores, agent_action_md,
                 agent_fix, manual_reward, state],
    )
    agent_btn.click(
        run_inference,
        inputs=[task_dropdown, state],
        outputs=[agent_reward, agent_scores, agent_action_md, agent_fix, state],
    )
    manual_btn.click(
        run_manual,
        inputs=[manual_line, manual_issues, manual_fix, state],
        outputs=[manual_reward, manual_scores, state],
    )


# ══════════════════════════════════════════════════════════════════════
# Mount Gradio inside FastAPI and launch
# ══════════════════════════════════════════════════════════════════════

app = gr.mount_gradio_app(api, gradio_app, path="/")


def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()