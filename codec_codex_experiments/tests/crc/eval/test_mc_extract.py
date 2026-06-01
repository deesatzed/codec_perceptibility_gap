from src.crc.eval.mc_extract import extract_choice, score_mc

OPTS = ["Paris", "Berlin", "Madrid", "Rome"]


def test_explicit_answer_is():
    assert extract_choice("After thinking, the answer is (C).", OPTS) == "C"


def test_qwen_trace_then_letter():
    assert extract_choice("Thinking Process:\n...\n</think>\n\nB", OPTS) == "B"


def test_option_text_substring():
    assert extract_choice("It is clearly Madrid.", OPTS) == "C"


def test_ambiguous_returns_none():
    # two different letters mentioned, no clear final answer
    assert extract_choice("Could be (A) or (B), unsure.", OPTS) is None


def test_score_mc_correct_and_extracted():
    ok, extracted = score_mc("the answer is (A)", OPTS, "A")
    assert ok is True and extracted is True


def test_score_mc_unextractable_is_wrong_and_flagged():
    ok, extracted = score_mc("hmm, no idea, maybe several", OPTS, "A")
    assert ok is False and extracted is False
