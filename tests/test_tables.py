"""End-to-end tests of chordutils.tables on a tiny synthetic MSK-CHORD study folder."""

import pandas as pd
import pytest

import chordutils as cu

HEADER = "#a\n#b\n#c\n#d\n"


def _write(folder, name, rows, header=""):
    d = pd.DataFrame(rows)
    (folder / name).write_text(header + d.to_csv(sep="\t", index=False))


@pytest.fixture
def study(tmp_path):
    _write(tmp_path, "data_clinical_sample.txt", [
        {"SAMPLE_ID": "A-T01", "PATIENT_ID": "A", "CANCER_TYPE": "X", "SAMPLE_TYPE": "Primary", "METASTATIC_SITE": None},
        {"SAMPLE_ID": "A-T02", "PATIENT_ID": "A", "CANCER_TYPE": "Other", "SAMPLE_TYPE": "Primary", "METASTATIC_SITE": None},
        {"SAMPLE_ID": "B-T01", "PATIENT_ID": "B", "CANCER_TYPE": "X", "SAMPLE_TYPE": "Metastasis", "METASTATIC_SITE": "Liver"},
    ], HEADER)
    _write(tmp_path, "data_clinical_patient.txt", [
        {"PATIENT_ID": "A", "OS_MONTHS": 10.0, "OS_STATUS": "1:DECEASED"},
        {"PATIENT_ID": "B", "OS_MONTHS": 20.0, "OS_STATUS": "0:LIVING"},
    ], HEADER)
    _write(tmp_path, "data_timeline_specimen.txt", [
        {"PATIENT_ID": "A", "START_DATE": 0, "SAMPLE_ID": "A-T01"},
        {"PATIENT_ID": "A", "START_DATE": 0, "SAMPLE_ID": "A-T02"},
        {"PATIENT_ID": "B", "START_DATE": 0, "SAMPLE_ID": "B-T01"},
    ])
    _write(tmp_path, "data_timeline_diagnosis.txt", [
        {"PATIENT_ID": "A", "START_DATE": -10, "DX_DESCRIPTION": "ADENOCARCINOMA | PANCREAS (M8140/3 | C250)", "SUMMARY": " Local "},
        {"PATIENT_ID": "A", "START_DATE": -500, "DX_DESCRIPTION": "ADENOCARCINOMA | COLON (M8140/3 | C180)", "SUMMARY": "Local"},
        {"PATIENT_ID": "B", "START_DATE": -100, "DX_DESCRIPTION": "ADENOCARCINOMA | PANCREAS (M8140/3 | C251)", "SUMMARY": "Distant"},
    ])
    _write(tmp_path, "data_timeline_surgery.txt", [
        {"PATIENT_ID": "A", "START_DATE": -10, "SUBTYPE": "PROCEDURE"},   # pancreas + colon sample same day
        {"PATIENT_ID": "A", "START_DATE": -200, "SUBTYPE": "PROCEDURE"},  # outside window
        {"PATIENT_ID": "B", "START_DATE": -120, "SUBTYPE": "SAMPLE"},    # 20 days before dx
    ])
    _write(tmp_path, "data_timeline_specimen_surgery.txt", [
        {"PATIENT_ID": "A", "START_DATE": -10, "SAMPLE_ID": "A-T01", "SEQ_DATE": 0},
        {"PATIENT_ID": "A", "START_DATE": -10, "SAMPLE_ID": "A-T02", "SEQ_DATE": 0},
        {"PATIENT_ID": "B", "START_DATE": -120, "SAMPLE_ID": "B-T01", "SEQ_DATE": 0},
    ])
    _write(tmp_path, "data_timeline_treatment.txt", [
        {"PATIENT_ID": "A", "START_DATE": 0, "STOP_DATE": 60, "SUBTYPE": "Chemo", "AGENT": "GEMCITABINE", "RX_INVESTIGATIVE": "N"},
        {"PATIENT_ID": "A", "START_DATE": 0, "STOP_DATE": 40, "SUBTYPE": "Chemo", "AGENT": "PACLITAXEL PROTEIN-BOUND", "RX_INVESTIGATIVE": "N"},
        {"PATIENT_ID": "A", "START_DATE": 100, "STOP_DATE": 103, "SUBTYPE": "Chemo", "AGENT": "CAPECITABINE", "RX_INVESTIGATIVE": "N"},
        {"PATIENT_ID": "A", "START_DATE": 200, "STOP_DATE": 215, "SUBTYPE": "Chemo", "AGENT": "CAPECITABINE", "RX_INVESTIGATIVE": "N"},
        {"PATIENT_ID": "A", "START_DATE": 300, "STOP_DATE": 900, "SUBTYPE": "Hormone", "AGENT": "LEUPROLIDE", "RX_INVESTIGATIVE": "N"},
        {"PATIENT_ID": "B", "START_DATE": -200, "STOP_DATE": -150, "SUBTYPE": "Chemo", "AGENT": "GEMCITABINE", "RX_INVESTIGATIVE": "N"},
    ])
    _write(tmp_path, "data_timeline_cancer_presence.txt", [
        {"PATIENT_ID": "A", "START_DATE": 50, "PROCEDURE_TYPE": "CT", "CHEST": True, "ABDOMEN": True, "PELVIS": True,
         "HEAD": False, "OTHER": False, "HAS_CANCER": "Y"},
        {"PATIENT_ID": "A", "START_DATE": 80, "PROCEDURE_TYPE": "MR", "CHEST": False, "ABDOMEN": False, "PELVIS": False,
         "HEAD": True, "OTHER": False, "HAS_CANCER": "Indeterminate"},
    ])
    _write(tmp_path, "data_timeline_progression.txt", [
        {"PATIENT_ID": "A", "START_DATE": 50, "PROCEDURE_TYPE": "CT", "PROGRESSION": "Y"},
    ])
    _write(tmp_path, "data_timeline_tumor_sites.txt", [
        {"PATIENT_ID": "A", "START_DATE": 50, "SOURCE_SPECIFIC": "CT", "TUMOR_SITE": "Liver"},
        {"PATIENT_ID": "A", "START_DATE": 50, "SOURCE_SPECIFIC": "CT", "TUMOR_SITE": "Lung"},
    ])
    return cu.ChordStudy(tmp_path)


CFG = cu.EntityConfig(
    "X", {"CANCER_TYPE": ["X"]}, cu.icdo_topography_pattern(["C25"]),
    windows={"surgery": (-90, None), "treatment": (0, None)},
    treatment=cu.TreatmentRules(subtypes=("Chemo",), min_row_span={"CAPECITABINE": 7},
                                min_episode_days=2, min_regimen_days={"CAPECITABINE": 21}),
)


@pytest.fixture
def cohort(study):
    return cu.select_cohort(study, CFG)


def test_cohort(cohort):
    assert cohort.samples["SAMPLE_ID"].tolist() == ["A-T01", "B-T01"]
    assert cohort.patients["OS_STATUS"].tolist() == [1, 0]
    assert cohort.diagnosis.set_index("PATIENT_ID")["START_DATE"].to_dict() == {"A": -10, "B": -100}
    assert cohort.diagnosis["SUMMARY"].tolist() == ["Local", "Distant"]


def test_treatment_episodes(cohort, study):
    tx = cu.treatment_episodes(cohort, study)
    # A: gem/nab-p 0-40, gem 41-60; short capecitabine record removed (span 3 < 7);
    #    capecitabine 200-215 removed (16 < 21 days); hormone subtype excluded.
    # B: therapy before diagnosis removed by the window.
    assert tx[["PATIENT_ID", "REGIMEN", "START_DATE", "STOP_DATE", "DAYS"]].values.tolist() == [
        ["A", "GEMCITABINE_PACLITAXEL PROTEIN-BOUND", 0, 40, 41],
        ["A", "GEMCITABINE", 41, 60, 20],
    ]
    assert tx["TIME_FROM_DIAGNOSIS"].tolist() == [10, 51]
    steps = cohort.log.to_frame().query("table == 'treatment'")["step"].tolist()
    assert "short CAPECITABINE records" in steps and "short CAPECITABINE episodes" in steps


def test_surgeries_prefer_cohort_sample(cohort, study):
    sx = cu.surgeries(cohort, study)
    assert sx[["PATIENT_ID", "START_DATE", "SAMPLE_ID", "SAMPLE_TYPE"]].values.tolist() == [
        ["A", -10, "A-T01", "Primary"],
        ["B", -120, "B-T01", "Metastasis"],
    ]
    assert sx["METASTATIC_SITE"].isna().tolist() == [True, False]
    assert sx["TIME_FROM_DIAGNOSIS"].tolist() == [0, -20]


def test_imaging_assessments(cohort, study):
    im = cu.imaging_assessments(cohort, study)
    assert im.values.tolist() == [["A", 50, "CT-TAB", "Y", "Y", "Liver,Lung"]]


def test_region_code():
    d = pd.DataFrame({"CHEST": [True, False], "ABDOMEN": [True, False], "PELVIS": [False, False],
                      "HEAD": [False, True], "OTHER": [False, False]})
    assert cu.tables.region_code(d).tolist() == ["TA", "H"]
