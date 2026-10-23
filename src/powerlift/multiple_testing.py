"""Multiple-comparison corrections for p-values.

Testing many metrics or many variants at level alpha each inflates the chance
of at least one false positive: with 20 independent true-null tests it is
``1 - 0.95^20 ~= 64%``. The corrections below adjust the p-values so that the
decision rule ``adjusted p <= alpha`` controls either

* the family-wise error rate (FWER, probability of any false positive):
  Bonferroni and Holm; or
* the false discovery rate (FDR, expected share of false positives among the
  rejections): Benjamini-Hochberg.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from powerlift._results import Result, format_p_value
from powerlift._validation import as_float_array, check_open_unit
from powerlift.exceptions import InvalidInputError

CorrectionMethod = Literal["bonferroni", "holm", "bh"]
_METHODS: tuple[CorrectionMethod, ...] = ("bonferroni", "holm", "bh")
_NAMES = {"bonferroni": "Bonferroni", "holm": "Holm", "bh": "Benjamini-Hochberg"}


@dataclass(frozen=True)
class MultipleTestingResult(Result):
    """Adjusted p-values and decisions for a family of tests.

    Attributes
    ----------
    method : {"bonferroni", "holm", "bh"}
        Correction that was applied.
    alpha : float
        FWER (Bonferroni, Holm) or FDR (BH) level.
    p_values : tuple of float
        Raw p-values, in the order given.
    adjusted_p_values : tuple of float
        Adjusted p-values, in the same order.
    reject : tuple of bool
        ``adjusted_p_value <= alpha`` for each hypothesis.
    """

    method: CorrectionMethod
    alpha: float
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    reject: tuple[bool, ...]

    def summary(self) -> str:
        """Return a human-readable summary."""
        target = "FDR" if self.method == "bh" else "FWER"
        lines = [
            f"{_NAMES[self.method]} correction, {target} controlled at {self.alpha:g}",
            "  test   raw p      adjusted p   reject",
        ]
        for i, (raw, adjusted, reject) in enumerate(
            zip(self.p_values, self.adjusted_p_values, self.reject, strict=True),
            start=1,
        ):
            lines.append(
                f"  {i:<6} {format_p_value(raw):<10} {format_p_value(adjusted):<12} "
                f"{'yes' if reject else 'no'}"
            )
        return "\n".join(lines)


def adjust_p_values(
    p_values: npt.ArrayLike,
    *,
    method: CorrectionMethod = "holm",
    alpha: float = 0.05,
) -> MultipleTestingResult:
    """Adjust p-values for multiple comparisons.

    Parameters
    ----------
    p_values : array_like
        Raw p-values of the family of tests.
    method : {"bonferroni", "holm", "bh"}, default "holm"
        * ``"bonferroni"``: ``min(1, m p_i)``. Controls FWER under any
          dependence.
        * ``"holm"``: step-down; with sorted ``p_(1) <= ... <= p_(m)``,
          ``adj_(i) = max_{j <= i} min(1, (m - j + 1) p_(j))``. Controls FWER
          under any dependence and is never less powerful than Bonferroni.
        * ``"bh"``: Benjamini-Hochberg step-up;
          ``adj_(i) = min_{j >= i} min(1, m p_(j) / j)``. Controls FDR when the
          tests are independent or positively dependent.
    alpha : float, default 0.05
        Level at which hypotheses are rejected (``adjusted p <= alpha``).

    Returns
    -------
    MultipleTestingResult
        Adjusted p-values and rejection decisions in the input order.

    Notes
    -----
    Matches ``statsmodels.stats.multitest.multipletests`` with methods
    ``"bonferroni"``, ``"holm"`` and ``"fdr_bh"``.

    References
    ----------
    Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2), 65-70.

    Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
    rate: a practical and powerful approach to multiple testing. *JRSS B*,
    57(1), 289-300.

    Examples
    --------
    >>> adjust_p_values([0.01, 0.04, 0.03], method="holm").adjusted_p_values
    (0.03, 0.06, 0.06)
    """
    p = as_float_array("p_values", p_values, min_size=1)
    if np.any((p < 0.0) | (p > 1.0)):
        raise InvalidInputError("p_values must lie between 0 and 1")
    if method not in _METHODS:
        raise InvalidInputError(f"method must be one of {_METHODS}, got {method!r}")
    alpha = check_open_unit("alpha", alpha)

    m = p.size
    order = np.argsort(p, kind="stable")
    sorted_p = p[order]
    if method == "bonferroni":
        adjusted_sorted = m * sorted_p
    elif method == "holm":
        # Step-down: the i-th smallest p-value is multiplied by m - i + 1 and
        # adjusted values are made non-decreasing in the sorted order.
        adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    else:
        # Step-up: multiply by m / i and take running minima from the largest.
        scaled = sorted_p * m / np.arange(1, m + 1)
        adjusted_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return MultipleTestingResult(
        method=method,
        alpha=alpha,
        p_values=tuple(p.tolist()),
        adjusted_p_values=tuple(adjusted.tolist()),
        reject=tuple(bool(r) for r in adjusted <= alpha),
    )
