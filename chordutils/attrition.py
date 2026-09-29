"""Bookkeeping of every filter step (attrition / CONSORT-style flow).

Each selection step in ``chordutils`` records what it did, why, and how many rows and
patients were left. The resulting table documents the cohort derivation for a methods
section or a flow diagram, and makes it obvious when a filter removes more than expected.
"""

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AttritionLog:
    """Ordered list of filter steps.

    Attributes
    ----------
    steps : list of dict
        One entry per step with ``table``, ``step``, ``reason``, ``rows_before``,
        ``rows_after``, ``patients_before``, ``patients_after`` and optional ``detail``.

    Examples
    --------
    >>> log = AttritionLog()
    >>> d = log.filter(d, d["AGE"] >= 18, table="patient", step="adults", reason="inclusion criterion")
    >>> log.to_frame()
    """

    id_col: str = "PATIENT_ID"
    steps: list = field(default_factory=list)

    def record(self, before: pd.DataFrame, after: pd.DataFrame, table: str, step: str,
               reason: str = "", detail: str = "") -> None:
        """Record a step from the DataFrames before and after it."""

        def n_patients(d):
            return d[self.id_col].nunique() if self.id_col in d.columns else pd.NA

        self.steps.append({"table": table, "step": step, "reason": reason,
                           "rows_before": len(before), "rows_after": len(after),
                           "patients_before": n_patients(before), "patients_after": n_patients(after),
                           "detail": detail})

    def filter(self, d: pd.DataFrame, mask, table: str, step: str, reason: str = "",
               detail: str = "") -> pd.DataFrame:
        """Apply a boolean mask to ``d``, record the step and return the filtered DataFrame."""

        out = d[mask]
        self.record(d, out, table, step, reason, detail)
        return out

    def note(self, d: pd.DataFrame, table: str, step: str, reason: str = "", detail: str = "") -> None:
        """Record a step that does not remove rows (e.g. a derived column or a check)."""

        self.record(d, d, table, step, reason, detail)

    def to_frame(self) -> pd.DataFrame:
        """All steps as a DataFrame, incl. the number of removed rows and patients."""

        cols = ["table", "step", "reason", "rows_before", "rows_after", "patients_before",
                "patients_after", "detail"]
        d = pd.DataFrame(self.steps, columns=cols)
        d.insert(5, "rows_removed", d["rows_before"] - d["rows_after"])
        d.insert(8, "patients_removed", d["patients_before"] - d["patients_after"])
        return d

    def _repr_html_(self):
        return self.to_frame()._repr_html_()

    def __repr__(self):
        return self.to_frame().drop(columns="detail").to_string()
