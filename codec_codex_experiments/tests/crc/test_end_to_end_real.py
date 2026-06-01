import pytest
from src.crc.panel import DecoderPanel, DEFAULT_FAMILIES
from src.crc.embedder import Embedder
from src.crc.run_crc import phase1_calibrate
from src.crc.bench import TINY_FACTUAL


@pytest.mark.heavy
def test_real_calibration_is_honest(tmp_path):
    panel = DecoderPanel(DEFAULT_FAMILIES[:2], cache_path=tmp_path / "c.json")
    cal = phase1_calibrate(panel, Embedder(), list(TINY_FACTUAL), TINY_FACTUAL)
    # Honest: either it proves the signal (VALID + threshold + P/R) or refuses.
    assert cal["verdict"] in {"VALID", "UNVALIDATED"}
    if cal["verdict"] == "VALID":
        assert cal["threshold"] is not None and "precision" in cal
    assert "hollow_agreement_warning" in cal
