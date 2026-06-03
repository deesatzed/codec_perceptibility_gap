"""The Verified | Refused contract and the Battery that composes controls.

A Control is a callable returning a CheckResult (passed + reason + evidence).
A Battery runs all controls; if ALL pass it returns Verified, else Refused with
the first failing reason and the full evidence. The default is to refuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)


# A Control is any zero-arg callable producing a CheckResult.
Control = Callable[[], CheckResult]


@dataclass(frozen=True)
class Verified:
    value: Any
    receipts: List[CheckResult]

    @property
    def status(self) -> str:
        return "VERIFIED"


@dataclass(frozen=True)
class Refused:
    reason: str                       # the first failing control's reason
    failed: str                       # the first failing control's name
    evidence: List[CheckResult]       # all checks (passed + failed)

    @property
    def status(self) -> str:
        return "REFUSED"


Result = Any  # Verified | Refused


class Battery:
    """Compose controls. evaluate(value) returns Verified(value) iff every control
    passes, else Refused with the first failure. Default posture = refuse."""

    def __init__(self, controls: List[Control]) -> None:
        if not controls:
            raise ValueError("a Battery needs at least one control")
        self._controls = controls

    def evaluate(self, value: Any = None) -> Result:
        receipts: List[CheckResult] = []
        first_fail: Optional[CheckResult] = None
        for ctrl in self._controls:
            r = ctrl()
            receipts.append(r)
            if not r.passed and first_fail is None:
                first_fail = r
        if first_fail is None:
            return Verified(value=value, receipts=receipts)
        return Refused(reason=first_fail.reason, failed=first_fail.name, evidence=receipts)
