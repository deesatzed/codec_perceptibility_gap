import pytest
from src.crc.panel import DecoderPanel, DEFAULT_FAMILIES


def test_family_constants_point_at_distinct_families():
    keys = [f["key"] for f in DEFAULT_FAMILIES]
    assert len(set(keys)) == len(keys) >= 2     # >=2 distinct families


@pytest.mark.heavy
def test_panel_answers_and_caches(tmp_path):
    panel = DecoderPanel(DEFAULT_FAMILIES[:2], cache_path=tmp_path / "c.json")
    a1 = panel.answer("What is the capital of France? One word.")
    assert len(a1) == 2 and all(isinstance(x, str) and x for x in a1.values())
    a2 = panel.answer("What is the capital of France? One word.")
    assert a1 == a2   # second call hits cache
