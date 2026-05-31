"""Flaw-catcher: cross-prompt replay attribution (ultrathink finding).

probe_text.distinction_key collapses many prompts into one region key (verified:
region (0,0,0,0) holds 10 unrelated prompts). On a trusted REPLAY the cascade
serves the cheap model WITHOUT auditing the current prompt, and books a saving
baselined on a DIFFERENT prompt's reference cost. So a saving can be emitted for
a route that was never verified on its own prompt.

The robust, nondeterminism-free invariant: NO saving may be booked for a route
that has no verdict for its own prompt. This test FAILS on the pre-fix runtime
(unaudited replay emits aps_savings>0 with verdict_passed is None) and PASSES once
the runtime audits every served-cheap prompt.

The logic companion substitutes ONLY the model I/O (MLXModel.generate) with tiny
deterministic stubs — the routing/graph/verifier logic under test is the REAL
code. This is a unit test of routing logic, not a mock of business logic (same
boundary CIO-II uses for its arbiter tests). The heavy test exercises the real
models end to end.
"""
import pytest

from src.dern_b.probe_text import distinction_key
from src.dern_b.mlx_backend import GenResult


# Two prompts proven to share a region key (the exploit precondition).
P_FIRST = "What is the capital of France? One word."
P_REPLAY = "What is the capital of Japan? One word."


def test_exploit_precondition_same_region_key():
    assert distinction_key(P_FIRST) == distinction_key(P_REPLAY)


class _StubModel:
    """Substitutes MLXModel.generate I/O only. Returns a fixed answer + realistic
    metering so the routing logic runs for real."""
    def __init__(self, answer, aps):
        self._answer = answer
        self._aps = aps
        self.active_params = 1_000_000

    def generate(self, prompt, max_tokens=64):
        return GenResult(text=self._answer, prompt_tokens=8, gen_tokens=4,
                         wall_seconds=0.1, active_params=self.active_params,
                         active_param_seconds=self._aps)


def _runtime_with_stubs(cheap_answer, ref_answer, audit_prob):
    from src.dern_b.runtime_b import DERNBRuntime
    rt = DERNBRuntime(epsilon=0.5, audit_prob=audit_prob, max_tokens=16, seed=0)
    rt.cheap = _StubModel(cheap_answer, aps=1.0e5)   # cheap: small aps
    rt.ref = _StubModel(ref_answer, aps=5.0e5)       # reference: larger aps
    return rt


def test_no_saving_without_a_per_prompt_verdict():
    """THE GATE. A route that books a positive saving must have a verdict for ITS
    OWN prompt. Pre-fix: an unaudited replay books aps_savings>0 with
    verdict_passed is None -> this assertion FAILS, catching the flaw."""
    # cheap == ref text so the first-encounter audit PASSES and mints region trust.
    rt = _runtime_with_stubs(cheap_answer="Paris", ref_answer="Paris", audit_prob=0.0)

    r1 = rt.route(P_FIRST)        # first encounter -> audited -> trust minted
    assert r1["audited"] is True

    r2 = rt.route(P_REPLAY)       # SAME region -> trusted replay; audit_prob=0 -> no audit

    # The invariant that must hold for the savings number to mean anything:
    booked_saving = r2.get("aps_savings", 0.0) > 0
    has_verdict = r2.get("verdict_passed") is not None
    assert not (booked_saving and not has_verdict), (
        "cross-prompt attribution: a saving was booked for a replay that was never "
        f"audited on its own prompt (verdict_passed={r2.get('verdict_passed')}, "
        f"aps_savings={r2.get('aps_savings')})"
    )


@pytest.mark.heavy
def test_real_models_replay_is_audited_on_its_own_prompt():
    """Heavy truth: with real models, a replayed prompt must be verified on its own
    text before any saving is booked."""
    from src.dern_b.runtime_b import DERNBRuntime
    rt = DERNBRuntime(epsilon=0.5, audit_prob=0.0, max_tokens=24, seed=0)
    r1 = rt.route(P_FIRST)
    r2 = rt.route(P_REPLAY)
    if r2.get("aps_savings", 0.0) > 0:
        assert r2.get("verdict_passed") is not None, (
            "real-model replay booked a saving with no per-prompt verdict"
        )
