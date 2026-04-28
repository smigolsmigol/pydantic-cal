"""pydantic_cal.crazy - paper-locked novel calibration method.

This subpackage exposes Federico's pre-registered ensemble-structure
calibration method from the f3d1 research line. It is gated behind a
runtime check until paper 1 publishes; importing it before the unlock
date raises ImportError with a pointer to the github release that will
flip the flag.

Once the paper is published, the gate unlocks via either:
  (a) a new release of pydantic-cal that ships the implementation, OR
  (b) the env var PYDANTIC_CAL_CRAZY_UNLOCK=1 set by Federico himself,
      for pre-publication co-author collaborators.

The public-knowledge calibration math (ECE, MCE, ACE, Brier, Fisher-Rao
distance, Jensen-Shannon, temperature/Platt/isotonic scaling, bootstrap
CIs, Murphy 1973 decomposition) ships in the parent package and is
unconditionally available. This subpackage adds ONE thing on top: the
ensemble-structure-aware scaler whose 3-5x ECE improvement is the
paper-1 headline result.
"""
from __future__ import annotations

import os

_UNLOCK_ENV = "PYDANTIC_CAL_CRAZY_UNLOCK"
_UNLOCK_REASON = (
    "pydantic_cal.crazy is paper-locked until f3d1 paper 1 publishes. "
    "Watch https://github.com/smigolsmigol/pydantic-cal/releases for the "
    "v0.1.0 unlock release. If you are a pre-publication co-author with "
    "the unlock token, set PYDANTIC_CAL_CRAZY_UNLOCK=1."
)


def _check_unlock() -> None:
    if os.environ.get(_UNLOCK_ENV, "").strip() not in {"1", "true", "yes"}:
        raise ImportError(_UNLOCK_REASON)


_check_unlock()

# Below this line: the gated implementation will land here at v0.1.0.
# Until then the import above raises and nothing follows is visible.
