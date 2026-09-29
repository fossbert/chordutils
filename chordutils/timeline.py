"""Generic operations on MSK-CHORD timeline tables.

Time axis: all START_DATE / STOP_DATE values in MSK-CHORD are days relative to the date of the
patient's first sequenced sample (day 0). Negative values are events before sequencing.
Analyses usually need a clinical time origin instead (diagnosis, start of therapy, ...);
``align_to_diagnosis`` adds the time relative to the diagnosis and restricts events to a window.
"""

from typing import Optional, Tuple

import pandas as pd

from .attrition import AttritionLog

Window = Tuple[Optional[int], Optional[int]]


def drop_constant_columns(d: pd.DataFrame, log: Optional[AttritionLog] = None,
                          table: str = "") -> pd.DataFrame:
    """Drop columns with at most one distinct non-missing value.

    Such columns carry no information within the cohort (e.g. CANCER_TYPE after selecting one
    cancer type, or GLEASON_* outside prostate cancer). Note that the dropped columns depend
    on the cohort; the names are recorded in ``log`` if given.
    """

    keep = d.nunique() > 1
    dropped = d.columns[~keep].tolist()
    if log is not None:
        log.note(d, table, "drop constant columns", "no information within the cohort",
                 detail=", ".join(dropped))
    return d.loc[:, keep]


def in_window(days: pd.Series, window: Window) -> pd.Series:
    """Boolean mask: ``lower <= days <= upper`` (bounds included, None = open)."""

    lo, hi = window
    mask = pd.Series(True, index=days.index)
    if lo is not None:
        mask &= days >= lo
    if hi is not None:
        mask &= days <= hi
    return mask


def _window_text(window: Window) -> str:
    lo, hi = window
    lo_txt = "-inf" if lo is None else str(lo)
    hi_txt = "+inf" if hi is None else str(hi)
    return f"[{lo_txt}, {hi_txt}] days from diagnosis"


def align_to_diagnosis(events: pd.DataFrame,
                       diagnosis: pd.DataFrame,
                       window: Window = (None, None),
                       time_col: str = "START_DATE",
                       dx_time_col: str = "START_DATE",
                       id_col: str = "PATIENT_ID",
                       out_col: str = "TIME_FROM_DIAGNOSIS",
                       log: Optional[AttritionLog] = None,
                       table: str = "") -> pd.DataFrame:
    """Add the time from diagnosis to each event and keep only events within a window.

    Parameters
    ----------
    events : pd.DataFrame
        A timeline table (one row per event) with ``id_col`` and ``time_col``.
    diagnosis : pd.DataFrame
        One row per patient with the day of diagnosis in ``dx_time_col``
        (e.g. ``Cohort.diagnosis``).
    window : (lower, upper)
        Events are kept if ``lower <= time from diagnosis <= upper`` (days, bounds included,
        None = open). E.g. ``(-90, None)`` keeps everything from 90 days before diagnosis
        onwards; ``(0, None)`` removes all events before the diagnosis.
    time_col, dx_time_col, id_col : str
        Column names of the event day, the diagnosis day and the patient identifier.
    out_col : str
        Name of the new column: event day minus diagnosis day (negative = before diagnosis).
    log : AttritionLog, optional
        If given, the removal of patients without diagnosis and the window filter are recorded.
    table : str
        Table name used in the log.

    Returns
    -------
    pd.DataFrame
        The rows of ``events`` within the window, all original columns plus ``out_col``.
        Events of patients without a diagnosis are removed.

    Raises
    ------
    ValueError
        If ``diagnosis`` has more than one row per patient (the time origin would be ambiguous
        and events would be duplicated by the merge).
    """

    if diagnosis[id_col].duplicated().any():
        raise ValueError("diagnosis must have one row per patient; resolve multiple diagnoses first")

    dx_day = diagnosis.set_index(id_col)[dx_time_col]
    has_dx = events[id_col].isin(dx_day.index)

    if log is not None:
        events = log.filter(events, has_dx, table, "patients with diagnosis",
                            "time origin (diagnosis) required")
    else:
        events = events[has_dx]

    events = events.assign(**{out_col: events[time_col] - events[id_col].map(dx_day)})
    mask = in_window(events[out_col], window)

    if log is not None:
        return log.filter(events, mask, table, "time window", _window_text(window))
    return events[mask]
