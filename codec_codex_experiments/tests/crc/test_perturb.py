from src.crc.perturb import (
    PERTURBED_ITEMS, perturbed_benchmark, memorized_traps,
)
from src.crc.correctness import is_correct


def test_items_have_distinct_answer_and_trap():
    # The whole design: the correct (premise) answer must differ from the
    # memorized trap, else there's no reasoning-vs-recall distinction.
    for it in PERTURBED_ITEMS:
        assert it.answer.lower() != it.memorized_trap.lower(), it.base_fact


def test_premise_answer_is_in_the_prompt():
    # The correct answer must be stated in the premise (so it's answerable by
    # reading, and the key is unambiguous ground truth).
    for it in PERTURBED_ITEMS:
        assert it.answer.lower() in it.prompt.lower(), it.prompt


def test_correctness_scorer_distinguishes_answer_from_trap():
    # A reasoning model (gives premise answer) scores correct; a parroting model
    # (gives memorized trap) scores wrong.
    for it in PERTURBED_ITEMS:
        assert is_correct(it.answer, it.answer) is True
        assert is_correct(it.memorized_trap, it.answer) is False


def test_benchmark_and_traps_well_formed():
    b = perturbed_benchmark()
    t = memorized_traps()
    assert len(b) == len(t) == len(PERTURBED_ITEMS) >= 30   # clears calibrate MIN_N
