"""Holdout budget: count the looks, shrink what is left per look, and offer a reusable-holdout (Thresholdout) alternative.

Evolution Lab used to re-read its 20% gate slice EVERY generation: any candidate that reaches the top has therefore been selected
partly on it, and the winner's holdout number is optimistically biased (winner's curse). Search Lab had no sealed slice at all.
Three tools fix this and are shared by both labs:

  ``split_three_way``   dev / gate / sealed chronological split (``sealed`` is read ONCE, after the run stops)
  ``HoldoutBudget``     a look counter that also shrinks the *usable* part of the gate slice with every look, and refuses to
                        open the sealed slice a second time
  ``Thresholdout``      Dwork et al.'s reusable holdout: answers with the TRAINING value unless it disagrees with the holdout by
                        more than a noisy threshold, and spends a finite budget when it has to reveal the holdout value

Everything is JSON-serialisable so a checkpoint can carry the counters across a resume.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class HoldoutExhausted(RuntimeError):
    """The look budget is spent (or the sealed slice was already read)."""


def split_three_way(df: pd.DataFrame, gate_frac: float = 0.2, sealed_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological (dev, gate, sealed) split: sealed = the LAST ``sealed_frac`` of bars, gate = the ``gate_frac`` before it, dev =
    the rest. Fractions are of the whole frame. Degenerate inputs never produce an empty dev slice."""
    n = len(df)
    if n < 10:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)
    sealed_n = int(n * max(0.0, float(sealed_frac)))
    gate_n = int(n * max(0.0, float(gate_frac)))
    dev_n = max(1, n - sealed_n - gate_n)
    dev = df.iloc[:dev_n].reset_index(drop=True)
    gate = df.iloc[dev_n:dev_n + gate_n].reset_index(drop=True)
    sealed = df.iloc[dev_n + gate_n:].reset_index(drop=True)
    return dev, gate, sealed


@dataclass
class HoldoutBudget:
    """Look counter + shrinking usable fraction for the gate slice, and a one-shot sealed slice.

    ``usable_fraction()`` is ``decay ** looks`` (never below ``floor``): the first look sees the whole gate slice, each later
    look sees a smaller, EARLIER part of it, so repeated reads cannot keep re-using the same most-recent bars. The sealed slice
    may be read exactly once per run."""

    max_looks: int = 200
    decay: float = 0.99
    floor: float = 0.5
    looks: int = 0
    sealed_reads: int = 0
    log: list = field(default_factory=list)

    def look(self, who: str = "") -> int:
        """Record one read of the gate slice. Raises ``HoldoutExhausted`` past ``max_looks``."""
        if self.looks >= int(self.max_looks):
            raise HoldoutExhausted(f"gate holdout look budget ({self.max_looks}) is spent")
        self.looks += 1
        if who:
            self.log.append({"look": self.looks, "who": str(who)})
            if len(self.log) > 500:
                self.log = self.log[-500:]
        return self.looks

    def usable_fraction(self) -> float:
        return float(max(self.floor, self.decay ** max(0, self.looks)))

    def usable_slice(self, gate_df: pd.DataFrame) -> pd.DataFrame:
        """The leading ``usable_fraction`` of the gate slice (earlier bars), at least 1 bar when the slice is non-empty."""
        if gate_df is None or not len(gate_df):
            return gate_df
        k = max(1, int(round(len(gate_df) * self.usable_fraction())))
        return gate_df.iloc[:k]

    def read_sealed(self, sealed_df: pd.DataFrame, who: str = "") -> pd.DataFrame:
        """Return the sealed slice and mark it read. A second read raises ``HoldoutExhausted``."""
        if self.sealed_reads >= 1:
            raise HoldoutExhausted("the sealed final slice has already been read once this run -- it is one-shot by design")
        self.sealed_reads += 1
        self.log.append({"look": "sealed", "who": str(who)})
        return sealed_df

    def to_dict(self) -> dict:
        return {"max_looks": self.max_looks, "decay": self.decay, "floor": self.floor, "looks": self.looks,
                "sealed_reads": self.sealed_reads, "log": list(self.log)}

    @classmethod
    def from_dict(cls, d: dict | None) -> "HoldoutBudget":
        d = dict(d or {})
        return cls(**{k: d[k] for k in ("max_looks", "decay", "floor", "looks", "sealed_reads", "log") if k in d})


class Thresholdout:
    """Dwork et al. (2015) reusable holdout for a bounded statistic.

    ``query(train_value, holdout_value)``: when the two agree within ``threshold`` (plus Laplace noise) the TRAINING value is
    returned and nothing is learned about the holdout; when they disagree the (noised) HOLDOUT value is returned and one unit of
    ``budget`` is spent. ``HoldoutExhausted`` once the budget is gone. The noise scale is ``sigma``; values are assumed to live
    on the scale of ``threshold`` (e.g. mean R in R units)."""

    def __init__(self, threshold: float = 0.05, sigma: float = 0.01, budget: int = 50, seed: int = 0):
        self.threshold = float(threshold)
        self.sigma = float(sigma)
        self.budget = int(budget)
        self.spent = 0
        self.queries = 0
        self._rng = np.random.default_rng(seed)

    def query(self, train_value: float, holdout_value: float) -> tuple[float, bool]:
        """(answer, revealed). ``revealed`` is True when the holdout value (noised) was returned."""
        if self.spent >= self.budget:
            raise HoldoutExhausted("Thresholdout budget exhausted")
        self.queries += 1
        gap = abs(float(train_value) - float(holdout_value))
        noisy_threshold = self.threshold + float(self._rng.laplace(0.0, 2.0 * self.sigma))
        if gap + float(self._rng.laplace(0.0, 4.0 * self.sigma)) > noisy_threshold:
            self.spent += 1
            return float(holdout_value) + float(self._rng.laplace(0.0, self.sigma)), True
        return float(train_value), False

    @property
    def remaining(self) -> int:
        return max(0, self.budget - self.spent)

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "sigma": self.sigma, "budget": self.budget, "spent": self.spent, "queries": self.queries}


def bonferroni_looks_alpha(alpha: float, looks: int) -> float:
    """The significance level a result read after ``looks`` looks at the same holdout must meet (Bonferroni over looks)."""
    return float(alpha) / max(1, int(looks))


def holdout_note(looks: int, max_looks: int) -> str:
    return (f"gate holdout read {looks} time(s) (budget {max_looks}); the winner's holdout number is selection-biased by every look -- "
            f"the sealed final slice, read once, is the unbiased check")


# ------------------------------------------------------------------------------------------------- persistent look registry
class LookRegistry:
    """Lifetime count of how many times each holdout slice (identified by its data fingerprint) has been read, across runs.

    A slice read 40 times by 40 different champions is no longer a holdout for the 41st. ``record`` returns the new count; callers
    report it and ``bonferroni_looks_alpha`` converts it into the significance level a result on that slice must meet."""

    def __init__(self, path=None):
        import json  # noqa: F401
        from pathlib import Path
        if path is None:
            from app.validation import trial_ledger
            path = Path(trial_ledger._PATH).parent / "holdout_looks.json"  # noqa: SLF001
        self.path = Path(path)

    def _load(self) -> dict:
        import json
        try:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    def count(self, slice_fp: str, kind: str = "gate") -> int:
        return int(self._load().get(f"{kind}:{slice_fp}", 0))

    def record(self, slice_fp: str, kind: str = "gate") -> int:
        import json
        d = self._load()
        k = f"{kind}:{slice_fp}"
        d[k] = int(d.get(k, 0)) + 1
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:  # noqa: BLE001 -- best-effort; the count returned is still right for this process
            pass
        return int(d[k])
