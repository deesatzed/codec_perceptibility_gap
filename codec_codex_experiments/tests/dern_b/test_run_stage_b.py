import pytest
from src.dern_b.run_stage_b import run_stage_b
from src.dern_b.prompts import FACTUAL


@pytest.mark.heavy
def test_stage_b_report_is_honest_and_safe():
    # small subset to keep the heavy run tractable
    report = run_stage_b(prompts=FACTUAL[:3], epsilon=0.5, max_tokens=12, seed=0)
    assert report["served_worse_than_reference"] == 0     # safety: never worse than ref
    assert report["manifest"]["tokens"] == "measured"
    assert report["manifest"]["joules"] in {"measured", "unavailable"}
    assert report["manifest"]["joules"] != "simulated"    # never fake
    assert 0.0 <= report["cheap_acceptance_rate"] <= 1.0
