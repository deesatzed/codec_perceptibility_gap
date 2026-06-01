"""Runner tests. The fake panel substitutes ONLY model I/O (canned answers); the
CRC logic under test (embed, disagreement, calibrate, verdict) is the real code.
Same boundary as the dern_b stubs."""
from src.crc.run_crc import phase1_calibrate, phase2_report
from src.crc.embedder import Embedder


class _FakePanel:
    def __init__(self, table):
        self.table = table

    def answer(self, prompt):
        return self.table[prompt]


def _bench():
    # 40 questions; 2 families AGREE on 'easy' (both right) and DISAGREE on 'hard'
    # (one right, one wrong) -> disagreement should track wrongness.
    table, key, qs = {}, {}, []
    for i in range(40):
        if i % 2 == 0:
            q = f"easy{i}"; table[q] = {"a": "Paris", "b": "Paris"}; key[q] = "Paris"
        else:
            q = f"hard{i}"; table[q] = {"a": "Paris", "b": "Lyon"}; key[q] = "Paris"
        qs.append(q)
    return qs, key, table


def test_phase1_returns_verdict_and_phase2_respects_it():
    qs, key, table = _bench()
    emb = Embedder()
    panel = _FakePanel(table)
    cal = phase1_calibrate(panel, emb, qs, key)
    assert cal["verdict"] in {"VALID", "UNVALIDATED"}

    rep = phase2_report(panel, emb, ["easy0", "hard1"], cal)
    assert len(rep["cards"]) == 2
    for card in rep["cards"]:
        if cal["verdict"] != "VALID":
            assert card["verdict"] == "uncalibrated"
        else:
            assert card["verdict"] in {"trust", "distrust", "out-of-calibration"}


def test_phase2_uncalibrated_never_claims_a_verdict():
    qs, key, table = _bench()
    emb = Embedder()
    fake_unvalidated = {"verdict": "UNVALIDATED", "threshold": None}
    rep = phase2_report(_FakePanel(table), emb, ["easy0", "hard1"], fake_unvalidated)
    assert all(c["verdict"] == "uncalibrated" for c in rep["cards"])
    assert rep["calibrated"] is False
