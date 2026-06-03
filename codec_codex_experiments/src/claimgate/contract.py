"""The earned-value contract: a Result that is Verified | Refused.

Design (first-principles, informed by Rust/Zig worldviews — see README):
- LAYER 1 (Rust Result): a Refused value is physically UNREACHABLE. You cannot
  read a number off a refusal by accident — you must unwrap()/expect() (which
  raises loudly on a refusal) or pattern-match. Ignoring a refusal takes effort.
- LAYER 2 (typestate): unwrapping a Verified yields a BRANDED value
  (require_verified()/Earned) so downstream functions can demand "earned" inputs
  and reject raw, unverified numbers.
- LAYER 3 (Zig explicit flow + comptime-spirit): Results compose via and_then/map
  that short-circuit a refusal explicitly through verify->transform->verify
  pipelines; and a Battery validates its OWN configuration at CONSTRUCTION (fail
  at build, not at evaluate).

A Control is a callable returning a CheckResult (passed + reason + evidence).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class RefusedError(Exception):
    """Raised when code tries to extract a value from a Refused result."""


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


# A Control is a zero-arg callable producing a CheckResult.
Control = Callable[[], CheckResult]


# ---- LAYER 2: the branded "earned" wrapper -------------------------------
@dataclass(frozen=True)
class Earned:
    """A value that PROVABLY passed a Battery. Downstream functions can require
    `Earned` in their signatures so a raw, unverified number cannot be passed in.
    Only `Verified.unwrap()` mints one; there is no public constructor path that
    bypasses a Battery."""
    value: Any
    _earned_brand: bool = field(default=True, repr=False)

    def get(self) -> Any:
        return self.value


# ---- LAYER 1 + 3: the Result sum type ------------------------------------
class Result:
    """Base for Verified | Refused. Carries the Rust-style extraction API and the
    Zig-style explicit pipeline combinators. Never instantiated directly."""
    status: str

    # --- Layer 1: enforced extraction ---
    def is_verified(self) -> bool:
        return isinstance(self, Verified)

    def is_refused(self) -> bool:
        return isinstance(self, Refused)

    def unwrap(self) -> Any:
        raise NotImplementedError

    def unwrap_or(self, default: Any) -> Any:
        raise NotImplementedError

    def expect(self, msg: str) -> Any:
        raise NotImplementedError

    def earned(self) -> Earned:
        """Layer 2: get the branded Earned value (raises on a refusal)."""
        raise NotImplementedError

    def match(self, *, verified: Callable[[Any], Any], refused: Callable[["Refused"], Any]) -> Any:
        """Force the caller to handle BOTH arms (Rust match)."""
        if isinstance(self, Verified):
            return verified(self.value)
        return refused(self)  # type: ignore[arg-type]

    # --- Layer 3: explicit, short-circuiting pipeline ---
    def and_then(self, fn: "Callable[[Any], Result]") -> "Result":
        """Run fn on the value if Verified; if Refused, short-circuit unchanged.
        fn must return a Result (verify -> transform -> verify chains)."""
        if isinstance(self, Verified):
            return fn(self.value)
        return self

    def map(self, fn: Callable[[Any], Any]) -> "Result":
        """Transform a Verified value, carrying receipts forward; Refused passes
        through unchanged. The transformed value is NOT re-verified (use and_then
        with a verifying step if the transform needs its own controls)."""
        if isinstance(self, Verified):
            return Verified(value=fn(self.value), receipts=self.receipts)
        return self


@dataclass(frozen=True)
class Verified(Result):
    value: Any
    receipts: List[CheckResult]
    status: str = field(default="VERIFIED")

    def unwrap(self) -> Any:
        return self.value

    def unwrap_or(self, default: Any) -> Any:
        return self.value

    def expect(self, msg: str) -> Any:
        return self.value

    def earned(self) -> Earned:
        return Earned(self.value)


@dataclass(frozen=True)
class Refused(Result):
    reason: str                       # the first failing control's reason
    failed: str                       # the first failing control's name
    evidence: List[CheckResult]       # all checks (passed + failed)
    status: str = field(default="REFUSED")

    # Layer 1: the value is UNREACHABLE. unwrap/expect raise; there is no .value.
    def unwrap(self) -> Any:
        raise RefusedError(f"refused by '{self.failed}': {self.reason}")

    def expect(self, msg: str) -> Any:
        raise RefusedError(f"{msg} [refused by '{self.failed}': {self.reason}]")

    def unwrap_or(self, default: Any) -> Any:
        return default

    def earned(self) -> Earned:
        raise RefusedError(f"cannot mint Earned from a refusal '{self.failed}': {self.reason}")

    def __getattr__(self, item: str):
        # trap accidental `.value` (and similar) access on a refusal -> loud, typed.
        if item == "value":
            raise RefusedError(
                f"refused result has no usable value (refused by '{self.failed}': "
                f"{self.reason}); use .unwrap_or(default) or .match(...)")
        raise AttributeError(item)


def require_verified(x: Any) -> Any:
    """Layer 2 boundary guard: accept an Earned (or a Verified) and return the raw
    value; reject anything else (a raw unverified number) with a loud error. Put
    this at the top of a function that must only consume earned numbers."""
    if isinstance(x, Earned):
        return x.value
    if isinstance(x, Verified):
        return x.value
    raise RefusedError(
        "unverified value passed where an Earned/Verified result is required; "
        "run it through a Battery first")


class Battery:
    """Compose controls. evaluate(value) -> Verified(value) iff every control
    passes, else Refused with the first failure. Default posture = refuse.

    Layer 3: configuration is validated at CONSTRUCTION — a malformed battery
    (empty, or a non-callable control) fails when you build it, not when you run
    it on data."""

    def __init__(self, controls: List[Control]) -> None:
        if not controls:
            raise ValueError("a Battery needs at least one control")
        for i, c in enumerate(controls):
            if not callable(c):
                raise TypeError(f"control #{i} is not callable: {c!r}")
        self._controls = list(controls)

    def evaluate(self, value: Any = None) -> Result:
        receipts: List[CheckResult] = []
        first_fail: Optional[CheckResult] = None
        for ctrl in self._controls:
            r = ctrl()
            if not isinstance(r, CheckResult):
                raise TypeError(f"control returned {type(r).__name__}, expected CheckResult")
            receipts.append(r)
            if not r.passed and first_fail is None:
                first_fail = r
        if first_fail is None:
            return Verified(value=value, receipts=receipts)
        return Refused(reason=first_fail.reason, failed=first_fail.name, evidence=receipts)
