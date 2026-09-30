"""Longitudinal values (tumour markers, ...) within lines of therapy, e.g. for tumgr.

``covariates.marker_kinetics`` summarises the marker course of a line (nadir, best change);
``marker_course`` keeps the individual values instead, one row per value and line, and
``to_tumgr`` converts them to the input of the tumour growth/decay models of the R package
tumgr (Wilkerson et al., Lancet Oncol 2017), called e.g. via ``cbrrwd.rbackend.tumor_growth.gdrate``.

Window rules (defaults; each is a parameter and every exclusion is logged)
---------------------------------------------------------------------------
1. Values from ``pre_days`` before the line start until ``post_days`` after the line end are
   considered, but only BEFORE the start of the next line (values from the next day of the
   next therapy on describe that therapy).
2. Before and on the day of the line start only the LAST value is kept: it is the baseline
   (tumgr models decay and regrowth from a baseline at time 0; a rising pre-treatment course
   would distort the fit). ``baseline='all'`` keeps every pre-treatment value.
3. Lines whose baseline is not above ``baseline_above`` (e.g. the upper limit of normal) are
   excluded: low baseline values (CA19-9 non-secretors, values reported as 0) carry no
   information on response.
4. Several values of one patient on the same day are combined (mean, median or first).
5. Time 0 is the baseline value (tumgr convention); days from the line start are kept as well.
   Each line gets its own identifier, so a patient contributes one course per line.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .attrition import AttritionLog

ID = "PATIENT_ID"
_SAME_DAY = {"mean": "mean", "median": "median", "first": "first"}


def _log_lines(log: Optional[AttritionLog], before: pd.DataFrame, after: pd.DataFrame, step: str,
               reason: str, table: str = "marker_course") -> None:
    """Record a step on the level of lines (rows = lines, patients = patients)."""

    if log is None:
        return
    def lines(d):
        return d[[ID, "LINE"]].drop_duplicates() if len(d) else pd.DataFrame(columns=[ID, "LINE"])
    log.record(lines(before), lines(after), table, step, reason)


def marker_course(lines: pd.DataFrame,
                  values: pd.DataFrame,
                  test: Optional[str] = None,
                  value_col: str = "RESULT",
                  test_col: str = "TEST",
                  time_col: str = "START_DATE",
                  pre_days: int = 21,
                  post_days: int = 21,
                  stop_at_next_line: bool = True,
                  baseline: str = "last",
                  baseline_above: Optional[float] = None,
                  min_values: int = 2,
                  same_day: str = "mean",
                  log: Optional[AttritionLog] = None) -> pd.DataFrame:
    """Values of a longitudinal variable within each line of therapy (see module docstring).

    Parameters
    ----------
    lines : pd.DataFrame
        One row per line with PATIENT_ID, LINE, LINE_START, LINE_END (e.g. a selection of
        ``endpoints.line_endpoints`` output such as first-line palliative FOLFIRINOX). For
        ``stop_at_next_line`` the column NEXT_LINE_START is used if present; it must refer to
        ALL lines of the patient, which ``line_endpoints`` provides. Without it, the next
        line is taken from ``lines`` itself, which is wrong if ``lines`` is a selection.
        Further columns (REGIMEN, SETTING, ...) are carried along.
    values : pd.DataFrame
        Measurements with PATIENT_ID, ``time_col`` (timeline day) and ``value_col``, e.g.
        db['TUMORMARKER'] or db['PERFORMANCE'].
    test : str, optional
        Keep only rows with ``values[test_col] == test`` (e.g. 'CA19-9 (U/mL)').
    pre_days, post_days : int
        Window from LINE_START - pre_days to LINE_END + post_days (both included).
    stop_at_next_line : bool
        Drop values on or after the start of the next line.
    baseline : {'last', 'all'}
        'last': only the last value up to and including the line start is kept (baseline);
        lines without such a value are excluded. 'all': all pre-treatment values in the
        window are kept and the first value of the course is time 0 (as in earlier analyses).
    baseline_above : float, optional
        Exclude lines whose baseline value is not strictly above this value (e.g. 37 U/mL
        for CA19-9). With ``baseline='all'`` the first value of the course is used.
    min_values : int
        Minimum number of values per line (baseline included).
    same_day : {'mean', 'median', 'first'}
        How several values of a patient on the same day are combined.
    log : AttritionLog, optional
        Every step is recorded with the number of lines and patients (e.g. ``cohort.log``).

    Returns
    -------
    pd.DataFrame
        One row per value and line: LINE_ID ('<PATIENT_ID>:L<LINE>'), PATIENT_ID, LINE, the
        carried-along line columns, DAY (timeline day), DAYS_FROM_LINE_START, DAYS_FROM_BASELINE
        (0 for the first value), VALUE, PHASE ('baseline', 'pre' or 'on_treatment'),
        N_SAME_DAY (number of values combined), ordered by line and day.
    """

    if baseline not in ("last", "all"):
        raise ValueError(f"baseline must be 'last' or 'all', got {baseline!r}")
    if same_day not in _SAME_DAY:
        raise ValueError(f"same_day must be one of {list(_SAME_DAY)}, got {same_day!r}")

    v = values if test is None else values[values[test_col] == test]
    v = v[[ID, time_col, value_col]].dropna(subset=[value_col])
    v = (v.groupby([ID, time_col])[value_col].agg([_SAME_DAY[same_day], "size"])
          .set_axis(["VALUE", "N_SAME_DAY"], axis=1).reset_index().rename(columns={time_col: "DAY"}))

    L = lines.copy()
    if "NEXT_LINE_START" not in L.columns:
        L["NEXT_LINE_START"] = L.sort_values([ID, "LINE_START"]).groupby(ID)["LINE_START"].shift(-1)
    L["LINE_ID"] = L[ID].astype(str) + ":L" + L["LINE"].astype(str)

    x = L.merge(v, on=ID)
    _log_lines(log, L, x, "lines with values", f"any {test or value_col} value of the patient")

    lo, hi = x["LINE_START"] - pre_days, x["LINE_END"] + post_days
    before = x
    x = x[(x["DAY"] >= lo) & (x["DAY"] <= hi)]
    _log_lines(log, before, x, "time window", f"[LINE_START - {pre_days}, LINE_END + {post_days}] days")

    if stop_at_next_line:
        before = x
        x = x[x["NEXT_LINE_START"].isna() | (x["DAY"] < x["NEXT_LINE_START"])]
        _log_lines(log, before, x, "before next line", "values from the start of the next line on removed")

    pre = x["DAY"] <= x["LINE_START"]
    if baseline == "last":
        last_pre = x[pre].groupby("LINE_ID")["DAY"].transform("max")
        keep = ~pre | (x["DAY"] == last_pre.reindex(x.index))
        before = x
        x = x[keep & x["LINE_ID"].isin(x.loc[pre, "LINE_ID"])]
        _log_lines(log, before, x, "baseline", f"last value up to line start; lines without one removed")
        x = x.assign(PHASE=np.where(x["DAY"] <= x["LINE_START"], "baseline", "on_treatment"))
    else:
        x = x.assign(PHASE=np.where(pre, "pre", "on_treatment"))

    x = x.sort_values(["LINE_ID", "DAY"], kind="stable")
    first = x.groupby("LINE_ID")["VALUE"].transform("first")
    if baseline_above is not None:
        before = x
        x = x[first > baseline_above]
        _log_lines(log, before, x, "baseline value", f"baseline > {baseline_above}")

    n = x.groupby("LINE_ID")["VALUE"].transform("size")
    before = x
    x = x[n >= min_values]
    _log_lines(log, before, x, "number of values", f">= {min_values} values per line")

    x = x.assign(DAYS_FROM_LINE_START=x["DAY"] - x["LINE_START"],
                 DAYS_FROM_BASELINE=x["DAY"] - x.groupby("LINE_ID")["DAY"].transform("min"))

    first_cols = ["LINE_ID", ID, "LINE"]
    tail = ["DAY", "DAYS_FROM_LINE_START", "DAYS_FROM_BASELINE", "VALUE", "PHASE", "N_SAME_DAY"]
    carried = [c for c in L.columns if c not in first_cols + tail]
    return x[first_cols + carried + tail].reset_index(drop=True)


def to_tumgr(course: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Input table for ``tumgr::gdrate`` from ``marker_course`` output.

    tumgr requires numeric input throughout, including the subject identifier, so every
    LINE_ID gets a running number (1, 2, ...).

    Returns
    -------
    (pd.DataFrame, dict)
        Table with columns name (int), date (days from baseline, int) and size (value), and
        the mapping {name: LINE_ID} to join the tumgr results back to the lines.
    """

    ids = pd.Index(course["LINE_ID"].unique())
    name = pd.Series(np.arange(1, len(ids) + 1), index=ids)
    out = pd.DataFrame({"name": course["LINE_ID"].map(name).astype(int),
                        "date": course["DAYS_FROM_BASELINE"].astype(int),
                        "size": course["VALUE"].astype(float)})
    return out.reset_index(drop=True), {int(k): v for k, v in zip(name.values, name.index)}
