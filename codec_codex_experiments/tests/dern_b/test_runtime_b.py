import pytest
from src.dern_b.runtime_b import DERNBRuntime


# Light prompt set: short factual prompts where the cheap model often agrees.
PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "What is 2 plus 2? Answer with just the number.",
    "Name the largest planet in our solar system. One word.",
]


@pytest.mark.heavy
def test_cascade_serves_only_cheap_verified_or_reference():
    rt = DERNBRuntime(epsilon=0.5, audit_prob=1.0, max_tokens=16, seed=0)
    for p in PROMPTS:
        rec = rt.route(p)
        # served is always either a verified-cheap or the reference authority
        assert rec["served"] in {"cheap", "reference"}
        if rec["served"] == "cheap":
            # a served cheap must have been audited-and-passed (Lane 2), not unverified
            assert rec["audited"] is True and rec["verdict_passed"] is True
        # measured dims are tagged measured; joules never measured
        assert rec["cost"]["_tags"]["total_tokens"] == "measured"
        assert rec["cost"]["_tags"]["active_param_seconds"] == "measured"
        assert rec["cost"]["_tags"]["joules"] != "measured"


@pytest.mark.heavy
def test_ledger_one_record_per_prompt():
    rt = DERNBRuntime(epsilon=0.5, audit_prob=1.0, max_tokens=16, seed=1)
    for p in PROMPTS:
        rt.route(p)
    assert len(rt.ledger) == len(PROMPTS)
