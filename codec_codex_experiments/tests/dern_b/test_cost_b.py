from src.dern_b.mlx_backend import GenResult
from src.dern_b.cost_b import cost_record, net_savings


def _gen(tok, secs, aps):
    return GenResult(text="x", prompt_tokens=tok // 2, gen_tokens=tok - tok // 2,
                     wall_seconds=secs, active_params=1_000_000, active_param_seconds=aps)


def test_measured_dims_tagged_measured_joules_never_measured():
    rec = cost_record(_gen(100, 1.0, 1e6), overhead_units=10.0)
    assert rec["_tags"]["total_tokens"] == "measured"
    assert rec["_tags"]["wall_seconds"] == "measured"
    assert rec["_tags"]["active_param_seconds"] == "measured"
    assert rec["_tags"]["joules"] == "unavailable"   # honest default
    assert rec["joules"] is None


def test_joules_can_be_derived_but_never_silently_measured():
    rec = cost_record(_gen(100, 1.0, 1e6), joules=42.0, joules_source="derived")
    assert rec["joules"] == 42.0 and rec["_tags"]["joules"] == "derived"


def test_net_savings_charges_overhead_to_aps_only():
    base = cost_record(_gen(200, 4.0, 4e6))
    cheap = cost_record(_gen(80, 0.5, 5e5), overhead_units=1000.0)
    # token savings ignore overhead; aps savings include it
    assert net_savings(base, cheap, "total_tokens") == base["total_tokens"] - cheap["total_tokens"]
    assert net_savings(base, cheap, "active_param_seconds") == round(
        base["active_param_seconds"] - (cheap["active_param_seconds"] + 1000.0), 4)
