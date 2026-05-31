from src.dern.graph import ExperienceGraph, Edge


def test_edge_written_only_after_pass():
    g = ExperienceGraph()
    g.record(key=(1, 2, 0), config={"name": "cheap"}, cost=2.0, proof="exact", passed=True)
    g.record(key=(9, 9, 9), config={"name": "mid"}, cost=3.0, proof="bounded", passed=False)
    assert g.lookup((1, 2, 0)) is not None
    assert g.lookup((9, 9, 9)) is None


def test_replay_returns_cheapest_trusted_edge():
    g = ExperienceGraph()
    g.record((1, 1, 1), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    g.record((1, 1, 1), {"name": "cheap"}, cost=2.0, proof="exact", passed=True)
    edge = g.lookup((1, 1, 1))
    assert edge.config["name"] == "cheap"
    assert edge.proof == "exact"


def test_fail_evicts_existing_trusted_edge_and_locks_region():
    g = ExperienceGraph()
    g.record((2, 2, 2), {"name": "cheap"}, cost=2.0, proof="bounded", passed=True)
    assert g.lookup((2, 2, 2)) is not None
    g.record((2, 2, 2), {"name": "cheap"}, cost=2.0, proof="bounded", passed=False)
    assert g.lookup((2, 2, 2)) is None
    assert g.is_locked((2, 2, 2)) is True


def test_trust_accumulates_on_repeated_bounded_pass():
    g = ExperienceGraph()
    g.record((3, 3, 3), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    g.record((3, 3, 3), {"name": "mid"}, cost=3.0, proof="bounded", passed=True)
    assert g.lookup((3, 3, 3)).trust >= 2
