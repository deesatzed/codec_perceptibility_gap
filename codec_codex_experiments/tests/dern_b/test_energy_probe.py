from src.dern_b.energy_probe import measure_energy, sudo_available


def test_energy_probe_never_fabricates_when_unavailable():
    # If sudo is not primed (the default in an unattended run), the probe MUST
    # return None joules and tag 'unavailable' — never a made-up number.
    result, joules, source = measure_energy(lambda: 7)
    assert result == 7
    if not sudo_available():
        assert joules is None
        assert source == "unavailable"
    else:
        # If a developer primed sudo, a measured value is allowed (>=0) tagged measured.
        assert source in {"measured", "unavailable"}
        if source == "measured":
            assert joules is not None and joules >= 0.0


def test_source_tag_is_honest_enum():
    _, _, source = measure_energy(lambda: None)
    assert source in {"measured", "unavailable"}
