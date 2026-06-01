"""Two-arm experiment tests. Fake panels substitute ONLY model I/O (canned
answers); the experiment logic (disagreement, calibration, difficulty control,
verdict) is the real code."""
import numpy as np
from src.crc.two_arm import disagreement_beyond_difficulty, run_two_arm
from src.crc.embedder import Embedder


def test_difficulty_control_detects_pure_proxy():
    # disagreement == difficulty exactly -> partial corr ~ 0 (no info beyond difficulty)
    rng = np.random.default_rng(0)
    difficulty = rng.uniform(0, 1, 100)
    disagree = difficulty.copy()
    wrong = (difficulty > 0.5).astype(float)
    out = disagreement_beyond_difficulty(disagree, wrong, difficulty)
    assert out["partial_corr_controlling_difficulty"] is None or abs(out["partial_corr_controlling_difficulty"]) < 0.2


def test_difficulty_control_detects_real_extra_signal():
    # disagreement carries info about wrongness that difficulty does NOT
    rng = np.random.default_rng(1)
    difficulty = rng.uniform(0, 0.4, 100)        # all low difficulty
    wrong = rng.integers(0, 2, 100).astype(float)
    disagree = wrong * 0.6 + rng.normal(0, 0.05, 100)  # tracks wrongness, not difficulty
    out = disagreement_beyond_difficulty(disagree, wrong, difficulty)
    assert out["partial_corr_controlling_difficulty"] > 0.2


class _FakePanel:
    def __init__(self, table):
        self.table = table

    def answer(self, prompt):
        return self.table[prompt]


def test_run_two_arm_hollow_agreement_when_models_parrot():
    # Arm 2: both families PARROT the memorized trap (agree + wrong) -> hollow agreement.
    from src.crc.perturb import PERTURBED_ITEMS
    emb = Embedder()
    arm2_key = {it.prompt: it.answer for it in PERTURBED_ITEMS}
    # both families emit the memorized trap (wrong, but identical -> low disagreement)
    table = {it.prompt: {"a": it.memorized_trap, "b": it.memorized_trap} for it in PERTURBED_ITEMS}
    # Arm 1: both families correct + agree
    arm1_key = {f"q{i}": "Paris" for i in range(32)}
    table.update({f"q{i}": {"a": "Paris", "b": "Paris"} for i in range(32)})
    out = run_two_arm(_FakePanel(table), emb, arm1_key, arm2_key)
    assert out["arm2_perturbed"]["ensemble_wrong_rate"] > 0.5      # models are wrong
    assert out["higher_concept_verdict"]["verdict"] in {"HOLLOW_AGREEMENT", "INCONCLUSIVE"}
