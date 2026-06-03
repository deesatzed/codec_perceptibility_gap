from src.crc.agent.tasks import TASKS, TOOL_LETTERS, render, difficulty_features


def test_every_task_has_valid_correct_tool():
    for t in TASKS:
        assert t.correct in TOOL_LETTERS, t.prompt_task


def test_has_both_easy_and_trap_tasks():
    traps = [t for t in TASKS if t.is_trap]
    easy = [t for t in TASKS if not t.is_trap]
    assert len(traps) >= 5 and len(easy) >= 5   # need a difficulty gradient + real errors


def test_difficulty_features_are_pre_action_only():
    # features must be derivable from the task text alone (no model output / correctness)
    for t in TASKS:
        f = difficulty_features(t)
        assert set(f) == {"task_len_words", "has_number", "mentions_file",
                          "mentions_recent", "is_question"}
        assert all(isinstance(v, float) for v in f.values())


def test_render_contains_all_tools_and_task():
    r = render(TASKS[0])
    for L in TOOL_LETTERS:
        assert f"{L})" in r
    assert "one letter" in r.lower()


def test_enough_tasks_for_stats():
    assert len(TASKS) >= 20
