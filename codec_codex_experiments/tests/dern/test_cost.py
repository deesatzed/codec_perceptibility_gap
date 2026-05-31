from src.dern.configs import COMPUTE_CONFIGS, full_config
from src.dern.cost import cost_vector, net_savings, ALL_SIMULATED


def test_cost_vector_dimensions_tagged_simulated_in_stage_a():
    v = cost_vector(full_config(), posture="default", overhead_units=0.1)
    assert set(v) >= {"compute", "thermal_slope", "dvfs", "idle", "carbon", "overhead", "_tags"}
    assert all(tag == "simulated" for tag in v["_tags"].values())


def test_low_fan_posture_cuts_thermal_cost():
    full = cost_vector(full_config(), posture="default", overhead_units=0.0)
    cool = cost_vector(full_config(), posture="low_fan", overhead_units=0.0)
    assert cool["thermal_slope"] < full["thermal_slope"]


def test_net_savings_includes_overhead():
    baseline = cost_vector(full_config(), posture="default", overhead_units=0.0)
    cheap = cost_vector(COMPUTE_CONFIGS[0], posture="low_fan", overhead_units=0.5)
    net = net_savings(baseline, cheap)
    assert net == (baseline["compute"] + baseline["thermal_slope"]) - (
        cheap["compute"] + cheap["thermal_slope"] + cheap["overhead"]
    )
