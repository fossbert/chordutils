"""Unit tests for io, config, attrition, timeline and cohort on small synthetic data."""

import pandas as pd
import pytest

import chordutils as cu
from chordutils.cohort import parse_os_status, select_diagnoses


# --- io ------------------------------------------------------------------------------------

CLINICAL = ("#Patient Identifier\tOverall Survival Status\n"
            "#Identifier to uniquely specify a patient.\tDESCRIPTION: vital status\n"
            "#STRING\tSTRING\n"
            "#1\t1\n"
            "PATIENT_ID\tOS_STATUS\n"
            "P-1\t1:DECEASED\n"
            "P-2\t0:LIVING #2\n")


def test_read_cbio_table_skips_meta_header_only(tmp_path):
    f = tmp_path / "data_clinical_patient.txt"
    f.write_text(CLINICAL)
    d = cu.read_cbio_table(f)
    assert d.columns.tolist() == ["PATIENT_ID", "OS_STATUS"]
    assert d["OS_STATUS"].tolist() == ["1:DECEASED", "0:LIVING #2"]  # '#' inside values survives


def test_clinical_dictionary(tmp_path):
    f = tmp_path / "data_clinical_patient.txt"
    f.write_text(CLINICAL)
    dd = cu.read_clinical_dictionary(f)
    assert dd.loc["OS_STATUS", "display_name"] == "Overall Survival Status"
    assert dd.loc["OS_STATUS", "description"] == "DESCRIPTION: vital status"


def test_study_timelines_and_copies(tmp_path):
    (tmp_path / "data_timeline_radiation.txt").write_text("PATIENT_ID\tSTART_DATE\nP-1\t5\n")
    study = cu.ChordStudy(tmp_path)
    assert study.timelines == ["radiation"]
    a = study.timeline("radiation")
    a.loc[0, "START_DATE"] = 99
    assert study.timeline("radiation").loc[0, "START_DATE"] == 5  # cache is not modified
    with pytest.raises(ValueError, match="Available: radiation"):
        study.timeline("surgery")


def test_parse_os_status():
    assert parse_os_status(pd.Series(["1:DECEASED", "0:LIVING"])).tolist() == [1, 0]
    s = parse_os_status(pd.Series(["1:DECEASED", None]))
    assert s.iloc[0] == 1 and pd.isna(s.iloc[1])


# --- config --------------------------------------------------------------------------------

@pytest.mark.parametrize("text, codes, hit", [
    ("ADENOCARCINOMA, NOS | PANCREAS, HEAD (M8140/3 | C250)", ["C25"], True),
    ("ADENOCARCINOMA, NOS | CECUM (M8140/3 | C180)", ["C18", "C19", "C20"], True),
    ("ADENOCARCINOMA, NOS | RECTUM, NOS (M8140/3 | C209)", ["C18", "C19", "C20"], True),
    ("ADENOCARCINOMA, NOS | LUNG, UPPER LOBE (M8140/3 | C341)", ["C25"], False),
    ("ADENOCARCINOMA, NOS | PROSTATE (M8140/3 | C619)", ["C61"], True),
    ("SOMETHING | C25 MENTIONED IN TEXT (M8140/3 | C619)", ["C25"], False),
])
def test_icdo_topography_pattern(text, codes, hit):
    assert bool(pd.Series([text]).str.contains(cu.icdo_topography_pattern(codes)).iloc[0]) is hit


def test_icdo_topography_pattern_rejects_non_codes():
    with pytest.raises(ValueError):
        cu.icdo_topography_pattern(["PANCREAS"])


def test_config_validation():
    with pytest.raises(ValueError):
        cu.EntityConfig("X", {}, "C25", multiple_diagnoses="last")
    with pytest.raises(ValueError):
        cu.EntityConfig("X", {}, "C25", windows={"surgery": (10, -10)})
    cfg = cu.EntityConfig("X", {}, "C25", windows={"surgery": (-90, None)})
    assert cfg.window("surgery") == (-90, None)
    assert cfg.window("radiation") == (None, None)


# --- attrition -----------------------------------------------------------------------------

def test_attrition_log_counts():
    d = pd.DataFrame({"PATIENT_ID": ["A", "A", "B", "C"], "x": [1, 2, 3, 4]})
    log = cu.AttritionLog()
    out = log.filter(d, d["x"] > 1, "t", "x > 1", "demo")
    assert len(out) == 3
    row = log.to_frame().iloc[0]
    assert (row.rows_before, row.rows_after, row.rows_removed) == (4, 3, 1)
    assert (row.patients_before, row.patients_after, row.patients_removed) == (3, 3, 0)


# --- timeline ------------------------------------------------------------------------------

DX = pd.DataFrame({"PATIENT_ID": ["A", "B"], "START_DATE": [100, -20]})
EV = pd.DataFrame({"PATIENT_ID": ["A", "A", "A", "B", "C"], "START_DATE": [0, 10, 100, 0, 5]})


def test_align_to_diagnosis_window_and_origin():
    log = cu.AttritionLog()
    out = cu.align_to_diagnosis(EV, DX, window=(-90, None), log=log, table="ev")
    assert out["TIME_FROM_DIAGNOSIS"].tolist() == [-90, 0, 20]   # A: 10-100, 100-100; B: 0-(-20)
    assert out.columns.tolist() == ["PATIENT_ID", "START_DATE", "TIME_FROM_DIAGNOSIS"]
    steps = log.to_frame()
    assert steps["rows_removed"].tolist() == [1, 1]                # C has no diagnosis; A day 0 outside


def test_align_to_diagnosis_bounds_included():
    out = cu.align_to_diagnosis(EV, DX, window=(0, 0))
    assert out["TIME_FROM_DIAGNOSIS"].tolist() == [0]


def test_align_to_diagnosis_requires_one_dx_per_patient():
    with pytest.raises(ValueError, match="one row per patient"):
        cu.align_to_diagnosis(EV, pd.concat([DX, DX]))


def test_drop_constant_columns():
    d = pd.DataFrame({"a": [1, 2], "b": [1, 1], "c": [None, None]})
    log = cu.AttritionLog(id_col="a")
    assert cu.drop_constant_columns(d, log, "t").columns.tolist() == ["a"]
    assert log.steps[0]["detail"] == "b, c"


# --- cohort --------------------------------------------------------------------------------

def _dx(rows):
    return pd.DataFrame(rows, columns=["PATIENT_ID", "START_DATE", "DX_DESCRIPTION"])


def test_select_diagnoses_first_and_error():
    dx = _dx([("A", 50, "X | BREAST (M8500/3 | C504)"),
              ("A", 10, "X | BREAST (M8500/3 | C509)"),
              ("A", 0, "X | LUNG (M8140/3 | C341)"),
              ("B", 5, "X | BREAST (M8500/3 | C504)")])
    cfg = cu.EntityConfig("BRCA", {}, cu.icdo_topography_pattern(["C50"]))
    log = cu.AttritionLog()
    out = select_diagnoses(dx, cfg, log)
    assert out.set_index("PATIENT_ID")["START_DATE"].to_dict() == {"A": 10, "B": 5}
    assert log.steps[-1]["detail"] == "1 patients with several matching diagnoses"

    strict = cu.EntityConfig("BRCA", {}, cfg.diagnosis_pattern, multiple_diagnoses="error")
    with pytest.raises(ValueError, match="1 patients"):
        select_diagnoses(dx, strict, cu.AttritionLog())
