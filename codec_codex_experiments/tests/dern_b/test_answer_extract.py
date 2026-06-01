"""Regression: extract_final_answer must handle each model family's reasoning-
trace format (gemma harmony channel, qwen </think>), plain text, and the
truncated-no-terminator case. Verbatim final segment only — never rewritten."""
from src.dern_b.mlx_backend import extract_final_answer


def test_qwen_think_close():
    assert extract_final_answer("Thinking Process:\n1. ...\n5. Paris.\n</think>\n\nParis") == "Paris"


def test_gemma_harmony_channel():
    assert extract_final_answer("<|channel>thought\nThe user asks X.<channel|>Paris") == "Paris"


def test_plain_text_unchanged():
    assert extract_final_answer("Paris") == "Paris"


def test_truncated_trace_returns_raw_not_empty():
    # no terminator reached (truncated) -> return the raw text, never empty/fabricated
    raw = "<|channel>thought\nreasoning only no close marker"
    assert extract_final_answer(raw) == raw


def test_takes_last_terminator_when_multiple():
    # if both a think-close and a channel-close appear, take the LAST boundary
    txt = "<|channel>thought\na<channel|>interim </think> FinalAnswer"
    assert extract_final_answer(txt) == "FinalAnswer"
