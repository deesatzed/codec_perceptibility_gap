"""Verifiable tool-selection tasks for the agent action-confidence skill-score
audit. Each task has ONE programmatically-correct tool (verifiable success, no
LLM judge), and difficulty features observable BEFORE the model acts.

Design intent: span easy (obvious tool) -> hard (the obvious tool is a trap, or
the task is ambiguous), so the agent genuinely errs and there is a real
difficulty gradient for the skill score to work against. Hand-authored,
human-verifiable ground truth (not generated).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

# fixed tool set for every task (stable letter mapping)
TOOLS = {
    "A": "calculator (arithmetic only)",
    "B": "web_search (current events / facts you don't know)",
    "C": "file_read (read a local file by path)",
    "D": "do_nothing (refuse / no tool needed)",
}
TOOL_LETTERS = ["A", "B", "C", "D"]


@dataclass(frozen=True)
class ToolTask:
    prompt_task: str          # the task description shown to the agent
    correct: str              # the verifiably-correct tool letter
    is_trap: bool             # True if the obvious tool is wrong (hard)
    note: str                 # provenance of the ground truth


def render(task: ToolTask) -> str:
    lines = ["You are an agent. Available tools:"]
    for L in TOOL_LETTERS:
        lines.append(f"  {L}) {TOOLS[L]}")
    lines.append(f"\nTask: {task.prompt_task}")
    lines.append("Which single tool do you call? Reply with one letter (A, B, C, or D).")
    return "\n".join(lines)


def difficulty_features(task: ToolTask) -> dict:
    """Features observable BEFORE acting (the skill-score reference forecast).
    Deliberately exclude anything derived from the model's answer or correctness."""
    words = task.prompt_task.split()
    return {
        "task_len_words": float(len(words)),
        "has_number": 1.0 if any(c.isdigit() for c in task.prompt_task) else 0.0,
        "mentions_file": 1.0 if (".txt" in task.prompt_task or "file" in task.prompt_task.lower() or "/" in task.prompt_task) else 0.0,
        "mentions_recent": 1.0 if any(w in task.prompt_task.lower() for w in ("today", "current", "latest", "now", "2026", "this year")) else 0.0,
        "is_question": 1.0 if "?" in task.prompt_task else 0.0,
    }


# Ground-truth verified by construction. Easy: obvious tool right. Trap: obvious wrong.
TASKS: List[ToolTask] = [
    # --- easy: arithmetic -> calculator ---
    ToolTask("Compute 47 * 83.", "A", False, "pure arithmetic"),
    ToolTask("What is 1024 divided by 8?", "A", False, "pure arithmetic"),
    ToolTask("Add 199 and 458.", "A", False, "pure arithmetic"),
    ToolTask("What is 15% of 240?", "A", False, "pure arithmetic"),
    ToolTask("Compute the square of 36.", "A", False, "pure arithmetic"),
    # --- easy: current/unknown facts -> web_search ---
    ToolTask("What is today's top news headline?", "B", False, "needs current info"),
    ToolTask("What is the current price of gold right now?", "B", False, "needs current info"),
    ToolTask("Who won the most recent World Cup this year?", "B", False, "needs current info"),
    ToolTask("What is the latest version of Python released?", "B", False, "needs current info"),
    # --- easy: read a file -> file_read ---
    ToolTask("Read the contents of /etc/hosts.", "C", False, "explicit file path"),
    ToolTask("Open the file report.txt and show its text.", "C", False, "explicit file"),
    ToolTask("What does the file /var/log/system.log contain?", "C", False, "explicit file"),
    # --- easy: nothing needed -> do_nothing ---
    ToolTask("Say hello to the user.", "D", False, "no tool needed"),
    ToolTask("Repeat the word 'apple'.", "D", False, "no tool needed"),
    ToolTask("Acknowledge that you understood.", "D", False, "no tool needed"),
    # --- TRAPS (hard): the obvious tool is wrong ---
    ToolTask("What is 2 plus 2? (You already know this; no tool needed.)", "D", True,
             "looks like calculator but trivially known -> do_nothing"),
    ToolTask("Compute the meaning of the file path /home/user. (It's just a path, not a calculation.)", "C", True,
             "has 'compute' (calculator trap) but is a file path"),
    ToolTask("Search your memory for the capital of France (a fact you know).", "D", True,
             "says 'search' but it's known -> do_nothing not web_search"),
    ToolTask("Read me the value of 9 times 9.", "A", True,
             "says 'read' (file trap) but it's arithmetic"),
    ToolTask("What is the current capital of France?", "D", True,
             "says 'current' (web trap) but capital is stable known fact"),
    ToolTask("Calculate which file is at /tmp/data.csv.", "C", True,
             "says 'calculate' but it's a file path"),
    ToolTask("Look up 100 minus 37 for me.", "A", True,
             "says 'look up' (web trap) but arithmetic"),
    ToolTask("Find today's date by reading the file calendar.txt.", "C", True,
             "says 'today' (web trap) but explicitly a file read"),
    ToolTask("Tell me 5 squared without any tools.", "D", True,
             "arithmetic but explicitly 'without tools' -> do_nothing"),
]
