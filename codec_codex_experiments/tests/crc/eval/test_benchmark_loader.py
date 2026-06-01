"""Loader tests. MMLU-Pro is cached after the approved fetch, so these run offline.
If the cache is absent (fresh clone), they skip rather than hit the network."""
import pytest
from src.crc.eval.benchmark_loader import load_items, to_answer_key, MCItem, NetworkApprovalRequired


def _cached():
    try:
        return load_items("mmlu_pro", allow_network=False, limit=5)
    except NetworkApprovalRequired:
        return None


def test_loader_offline_or_skip():
    items = _cached()
    if items is None:
        pytest.skip("MMLU-Pro not cached; needs approved fetch")
    assert all(isinstance(i, MCItem) for i in items)
    for it in items:
        assert len(it.options) >= 2
        assert it.answer_letter and it.answer_letter.isalpha()
        assert it.question.strip().endswith("letter of the correct option.")


def test_answer_key_shape():
    items = _cached()
    if items is None:
        pytest.skip("MMLU-Pro not cached")
    key = to_answer_key(items)
    assert len(key) == len(items)
    assert all(isinstance(v, str) for v in key.values())


def test_uncached_unapproved_raises():
    # a nonsense benchmark name with no cache + no network must refuse, not fetch
    with pytest.raises((NetworkApprovalRequired, KeyError)):
        load_items("does_not_exist", allow_network=False)
