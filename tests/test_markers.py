"""Tests for chordutils.markers (values within lines, tumgr input)."""

import numpy as np
import pandas as pd
import pytest

from chordutils.attrition import AttritionLog
from chordutils.markers import marker_course, to_tumgr

LINES = pd.DataFrame({"PATIENT_ID": ["A", "A", "B", "C"], "LINE": [1, 2, 1, 1],
                      "LINE_START": [0, 150, 0, 0], "LINE_END": [100, 300, 60, 60],
                      "REGIMEN": ["FOLFIRINOX", "GEM/NAB-P", "FOLFIRINOX", "FOLFIRINOX"]})


def mk(rows, test="CA19-9 (U/mL)"):
    return pd.DataFrame([(p, d, v, test) for p, d, v in rows], columns=["PATIENT_ID", "START_DATE", "RESULT", "TEST"])


VALUES = mk([
    ("A", -40, 50.0),                    # outside pre window (21 days)
    ("A", -10, 400.0),                   # pre-treatment, not the last one
    ("A", -2, 500.0),                    # baseline of line 1
    ("A", 30, 300.0), ("A", 30, 200.0), ("A", 30, 100.0),   # same day: mean 200, median 200 / first 300
    ("A", 60, 150.0),
    ("A", 115, 140.0),                   # after line end, within post window (<= 121)
    ("A", 130, 900.0),                   # after post window
    ("A", 150, 1000.0),                  # start of line 2: baseline of line 2, not in line 1
    ("A", 200, 800.0),
    ("B", -5, 20.0), ("B", 30, 10.0),    # low baseline
    ("C", 10, 300.0), ("C", 40, 200.0),  # no baseline before line start
]) 


def test_default_rules():
    c = marker_course(LINES, VALUES, "CA19-9 (U/mL)")
    l1 = c[c.LINE_ID == "A:L1"]
    assert l1["DAY"].tolist() == [-2, 30, 60, 115]
    assert l1["VALUE"].tolist() == [500, 200, 150, 140]
    assert l1["PHASE"].tolist() == ["baseline", "on_treatment", "on_treatment", "on_treatment"]
    assert l1["DAYS_FROM_BASELINE"].tolist() == [0, 32, 62, 117]
    assert l1["DAYS_FROM_LINE_START"].tolist() == [-2, 30, 60, 115]
    assert l1["N_SAME_DAY"].tolist() == [1, 3, 1, 1]
    assert l1["REGIMEN"].unique().tolist() == ["FOLFIRINOX"]
    l2 = c[c.LINE_ID == "A:L2"]
    assert l2["DAY"].tolist() == [150, 200] and l2["PHASE"].iloc[0] == "baseline"
    assert "C:L1" not in set(c.LINE_ID)             # no baseline -> excluded
    assert "B:L1" in set(c.LINE_ID)                 # no baseline threshold by default


def test_same_day_options():
    med = marker_course(LINES, VALUES, "CA19-9 (U/mL)", same_day="median")
    first = marker_course(LINES, VALUES, "CA19-9 (U/mL)", same_day="first")
    day30 = lambda c: c.query("LINE_ID == 'A:L1' and DAY == 30")["VALUE"].item()
    assert (day30(med), day30(first)) == (200.0, 300.0)
    with pytest.raises(ValueError):
        marker_course(LINES, VALUES, same_day="max")


def test_next_line_stops_course():
    # a long post window would reach day 150 (next line start) and later
    c = marker_course(LINES, VALUES, "CA19-9 (U/mL)", post_days=200)
    assert c.query("LINE_ID == 'A:L1'")["DAY"].max() == 130
    c2 = marker_course(LINES, VALUES, "CA19-9 (U/mL)", post_days=200, stop_at_next_line=False)
    assert c2.query("LINE_ID == 'A:L1'")["DAY"].max() == 200


def test_next_line_start_from_full_table():
    sel = LINES[LINES.LINE == 1].copy()              # selection loses line 2 ...
    sel["NEXT_LINE_START"] = [150, np.nan, np.nan]  # ... unless NEXT_LINE_START is provided
    c = marker_course(sel, VALUES, "CA19-9 (U/mL)", post_days=200)
    assert c.query("LINE_ID == 'A:L1'")["DAY"].max() == 130


def test_baseline_threshold_min_values_and_log():
    log = AttritionLog()
    c = marker_course(LINES, VALUES, "CA19-9 (U/mL)", baseline_above=37, min_values=3, log=log)
    assert set(c.LINE_ID) == {"A:L1"}              # B: baseline 20; A:L2 only 2 values
    steps = log.to_frame().set_index("step")
    assert steps.loc["baseline value", "rows_removed"] == 1
    assert steps.loc["number of values", "rows_removed"] == 1
    assert steps.loc["baseline", "rows_removed"] == 1   # C without baseline


def test_baseline_all():
    c = marker_course(LINES, VALUES, "CA19-9 (U/mL)", baseline="all")
    l1 = c[c.LINE_ID == "A:L1"]
    assert l1["DAY"].tolist()[:2] == [-10, -2] and l1["PHASE"].tolist()[:2] == ["pre", "pre"]
    assert l1["DAYS_FROM_BASELINE"].iloc[0] == 0
    assert "C:L1" in set(c.LINE_ID)                 # first value may be on treatment


def test_generic_value_column():
    ecog = pd.DataFrame({"PATIENT_ID": "A", "START_DATE": [-3, 40, 90], "ECOG": [1, 1, 2]})
    c = marker_course(LINES, ecog, value_col="ECOG")
    assert c.query("LINE_ID == 'A:L1'")["VALUE"].tolist() == [1, 1, 2]


def test_to_tumgr():
    c = marker_course(LINES, VALUES, "CA19-9 (U/mL)", baseline_above=37)
    t, names = to_tumgr(c)
    assert list(t.columns) == ["name", "date", "size"]
    assert all(pd.api.types.is_numeric_dtype(t[col]) for col in t.columns)
    assert set(names.values()) == set(c.LINE_ID)
    first = t[t.name == next(k for k, v in names.items() if v == "A:L1")]
    assert first["date"].tolist() == [0, 32, 62, 117] and first["size"].tolist() == [500, 200, 150, 140]
