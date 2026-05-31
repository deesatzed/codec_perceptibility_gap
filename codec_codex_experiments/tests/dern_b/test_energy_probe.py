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


def test_coverage_guard_refuses_to_stretch_sparse_samples(monkeypatch):
    # If powermetrics captured only a tiny slice of a long work window, the probe
    # MUST return 'unavailable' (joules None) rather than stretching avg power
    # over the window. Exercises the real measure_energy coverage branch.
    import time as _t
    import src.dern_b.energy_probe as ep

    monkeypatch.setattr(ep, "sudo_available", lambda: True)
    # idle baseline: a couple of recognizable samples
    monkeypatch.setattr(ep, "_run_powermetrics",
                        lambda n, i: "CPU Power: 200 mW\nGPU Power: 0 mW\n" * n)
    # Stub the file-based powermetrics Popen so no real sudo is needed: we make the
    # captured file contain only 2 samples while fn sleeps long enough that 2
    # samples (at interval) cover < 60% of the work window.

    class _FakeProc:
        pid = 99999
        def __init__(self, *a, **k):
            # write 2 samples into the stdout file handle
            fh = k.get("stdout")
            if fh is not None:
                fh.write("CPU Power: 7000 mW\nGPU Power: 0 mW\n"
                         "CPU Power: 7000 mW\nGPU Power: 0 mW\n")
        def terminate(self): pass
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(ep.subprocess, "Popen", _FakeProc)

    def slow_fn():
        _t.sleep(1.2)   # work window ~1.2s; 2 samples @200ms = 0.4s sampled => 33% < 60%
        return "done"

    result, reading = ep.measure_energy(slow_fn, interval_ms=200, idle_samples=2)
    assert result == "done"
    assert reading.source == "unavailable"
    assert reading.joules_total is None        # never stretched/fabricated
    assert "coverage" in reading.note.lower()
