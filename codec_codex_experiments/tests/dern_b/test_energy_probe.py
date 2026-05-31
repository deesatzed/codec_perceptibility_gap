from src.dern_b.energy_probe import (
    measure_energy, sudo_available, _component_power_samples, EnergyReading,
)


def test_energy_probe_never_fabricates_when_unavailable():
    # If sudo is not primed (the default unattended run), the probe MUST return
    # None joules tagged 'unavailable' — never a made-up number.
    result, reading = measure_energy(lambda: 7)
    assert result == 7
    assert isinstance(reading, EnergyReading)
    if not sudo_available():
        assert reading.joules_total is None
        assert reading.source == "unavailable"
    else:
        assert reading.source in {"measured", "unavailable"}
        if reading.source == "measured":
            assert reading.joules_total is not None and reading.joules_total >= 0.0


def test_source_tag_is_honest_enum():
    _, reading = measure_energy(lambda: None)
    assert reading.source in {"measured", "unavailable"}


def test_parser_sums_components_and_ignores_combined():
    # Two samples; each has CPU+GPU component lines and a 'Combined Power' total
    # that MUST be ignored (else we double-count).
    text = """
*** Sampled system activity ***
CPU Power: 1000 mW
GPU Power: 500 mW
Combined Power (CPU + GPU + ANE): 1500 mW
*** Sampled system activity ***
CPU Power: 2000 mW
GPU Power: 1000 mW
Combined Power (CPU + GPU + ANE): 3000 mW
""".strip()
    samples = _component_power_samples(text)
    # sample 1 = 1.0 + 0.5 = 1.5 W ; sample 2 = 2.0 + 1.0 = 3.0 W
    assert samples == [1.5, 3.0]


def test_parser_handles_real_m4_output_with_stray_gpu_lines():
    # VERBATIM real M4 Pro powermetrics output: blocks start on CPU Power, and
    # powermetrics interleaves STRAY 'GPU Power' lines outside the block that must
    # NOT create phantom samples. Only two real samples here.
    real = """
CPU Power: 146 mW
GPU Power: 6 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 152 mW
GPU Power: 9 mW
CPU Power: 77 mW
GPU Power: 9 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 86 mW
GPU Power: 6 mW
""".strip()
    samples = _component_power_samples(real)
    # sample 1 = 146+6+0 = 0.152 W ; sample 2 = 77+9+0 = 0.086 W. Stray GPU lines ignored.
    assert samples == [0.152, 0.086]


def test_parser_returns_empty_on_unrecognized_format():
    # Unknown format -> no samples -> caller returns 'unavailable', never a guess.
    assert _component_power_samples("garbage\nno power lines here\n") == []
