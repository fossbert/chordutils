"""Unit tests for chordutils.pivot_therapies on small synthetic timelines.

Each test documents one rule of the algorithm (see the docstring of pivot_therapies).
"""

import numpy as np
import pandas as pd
import pytest

import chordutils as cu


def rows(*specs, investigative=None):
    """Build a treatment timeline from (agent, start, stop) tuples."""

    d = pd.DataFrame(specs, columns=["AGENT", "START_DATE", "STOP_DATE"])
    d["RX_INVESTIGATIVE"] = "N" if investigative is None else investigative
    d["SUBTYPE"] = "Chemo"
    return d


def episodes(d, **kw):
    """pivot_therapies output as a list of (agent, start, stop) tuples."""

    out = cu.pivot_therapies(d, **kw)
    return [(r.agent, r.start, r.stop) for r in out.itertuples()]


def assert_no_overlap(out):
    s = out.sort_values("start")
    assert (s["start"].to_numpy()[1:] > s["stop"].to_numpy()[:-1]).all()


def test_single_agent():
    assert episodes(rows(("GEMCITABINE", 0, 100))) == [("GEMCITABINE", 0, 100)]


def test_days_include_both_boundaries():
    out = cu.pivot_therapies(rows(("GEMCITABINE", 10, 19)))
    assert out["days"].tolist() == [10]


def test_identical_ranges_form_one_combination():
    d = rows(("OXALIPLATIN", 0, 50), ("FLUOROURACIL", 0, 50), ("LEUCOVORIN", 0, 50))
    assert episodes(d) == [("FLUOROURACIL_LEUCOVORIN_OXALIPLATIN", 0, 50)]


def test_partial_overlap_prunes_existing():
    # Rule 2, one consecutive remainder: gem alone after nab-paclitaxel stops
    d = rows(("GEMCITABINE", 0, 100), ("PACLITAXEL PROTEIN-BOUND", 0, 80))
    assert episodes(d) == [("GEMCITABINE_PACLITAXEL PROTEIN-BOUND", 0, 80), ("GEMCITABINE", 81, 100)]


def test_new_agent_extends_beyond_existing():
    # Rule 3: remainder of the new agent becomes its own episode
    d = rows(("OXALIPLATIN", 0, 30), ("FLUOROURACIL", 10, 60))
    assert episodes(d) == [("OXALIPLATIN", 0, 9), ("FLUOROURACIL_OXALIPLATIN", 10, 30), ("FLUOROURACIL", 31, 60)]


def test_split_keeps_long_blocks():
    # Rule 2, several blocks: capecitabine interrupted by a short course of oxaliplatin
    d = rows(("CAPECITABINE", 0, 100), ("OXALIPLATIN", 40, 50))
    assert episodes(d, min_size=7) == [("CAPECITABINE", 0, 39), ("CAPECITABINE_OXALIPLATIN", 40, 50),
                                       ("CAPECITABINE", 51, 100)]


def test_split_discards_short_blocks():
    # Remainders of 3 days (0-2) are shorter than min_size and are dropped
    d = rows(("CAPECITABINE", 0, 100), ("OXALIPLATIN", 3, 90))
    assert episodes(d, min_size=7) == [("CAPECITABINE_OXALIPLATIN", 3, 90), ("CAPECITABINE", 91, 100)]


def test_single_short_remainder_is_kept():
    # A single consecutive remainder is kept irrespective of min_size (documented asymmetry)
    d = rows(("GEMCITABINE", 0, 100), ("CISPLATIN", 0, 97))
    assert episodes(d, min_size=7) == [("CISPLATIN_GEMCITABINE", 0, 97), ("GEMCITABINE", 98, 100)]


def test_sequential_agents_do_not_combine():
    d = rows(("GEMCITABINE", 0, 50), ("FLUOROURACIL", 60, 100))
    assert episodes(d) == [("GEMCITABINE", 0, 50), ("FLUOROURACIL", 60, 100)]


def test_investigational_label():
    d = rows(("GEMCITABINE", 0, 50), ("INVESTIGATIVE", 0, 50), investigative=["N", "Y"])
    assert episodes(d) == [("GEMCITABINE_INVESTIGATIVE:Chemo", 0, 50)]


def test_many_rows_leave_no_digits_in_names():
    # Regression for the legacy bug: with >= 10 rows the row index had 2 digits and
    # the old regex left one digit behind (e.g. 'OXALIPLATIN1').
    specs = [("FLUOROURACIL", 28 * k, 28 * k + 2) for k in range(12)]
    specs += [("OXALIPLATIN", 28 * 11, 28 * 11 + 1), ("GEMCITABINE", 400, 500), ("CAPECITABINE", 420, 430)]
    out = cu.pivot_therapies(rows(*specs), min_size=7)
    assert not out["agent"].str.contains(r"\d").any(), out["agent"].tolist()
    assert "FLUOROURACIL_OXALIPLATIN" in out["agent"].tolist()


def test_agent_names_ending_in_digits_are_preserved():
    d = rows(("SODIUM IODIDE I-131", 0, 10), ("AC225 H11B6 MED 20-321", 5, 20))
    assert episodes(d) == [("SODIUM IODIDE I-131", 0, 4), ("AC225 H11B6 MED 20-321_SODIUM IODIDE I-131", 5, 10),
                           ("AC225 H11B6 MED 20-321", 11, 20)]


def test_row_order_does_not_matter():
    # Legacy code produced overlapping episodes for unsorted input (B returned as 1-30).
    d = rows(("A", 10, 20), ("B", 1, 30))
    out = cu.pivot_therapies(d)
    assert episodes(d) == [("B", 1, 9), ("A_B", 10, 20), ("B", 21, 30)]
    assert_no_overlap(out)
    assert episodes(d.iloc[::-1]) == episodes(d)


def test_uncovered_days_are_counted_once():
    # Total treated days (union of all agent ranges) is preserved when nothing is discarded
    d = rows(("A", 0, 60), ("B", 20, 40), ("C", 30, 90), ("D", 100, 120))
    out = cu.pivot_therapies(d, min_size=1)
    assert_no_overlap(out)
    union = set().union(*[range(s, e + 1) for _, s, e in d[["AGENT", "START_DATE", "STOP_DATE"]].itertuples(index=False)])
    assert out["days"].sum() == len(union)


def test_numpy_integers_accepted():
    d = rows(("GEMCITABINE", 0, 10)).astype({"START_DATE": np.int64, "STOP_DATE": np.int64})
    assert episodes(d) == [("GEMCITABINE", 0, 10)]


def test_empty_input():
    out = cu.pivot_therapies(rows())
    assert out.empty and list(out.columns) == ["agent", "start", "stop", "days"]


def test_validate_event_accepts_numpy_ints():
    assert cu.validate_event(np.int64(5))
    assert cu.validate_event((np.int64(1), 10))
    assert not cu.validate_event(True)
    assert not cu.validate_event((1, 2, 3))
    with pytest.raises(ValueError, match="got 'x'"):
        cu.gen_time_range("x", 0)


def test_gen_time_range():
    assert cu.gen_time_range(10, -3).tolist() == [7, 8, 9, 10]
    assert cu.gen_time_range(10, 2).tolist() == [10, 11, 12]
    assert cu.gen_time_range((10, 12), 1).tolist() == [9, 10, 11, 12, 13]
