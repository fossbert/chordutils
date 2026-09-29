"""Unit tests for lines, endpoints, survival and covariates on small synthetic data."""

import numpy as np
import pandas as pd
import pytest

import chordutils as cu
from chordutils.config import LineRules
from chordutils.covariates import (baseline_values, column_label, genomic_features, line_history,
                                   marker_kinetics, parse_dx_description)
from chordutils.endpoints import collapse_assessments, line_endpoints, os_base, patient_followup
from chordutils.lines import build_lines_of_therapy, classify_regimen
from chordutils.survival import benjamini_hochberg, km_estimate, landmark, to_counting_process

RULES = LineRules(
    ignored_agents=("LEUCOVORIN",),
    equivalent_agents={"CAPECITABINE": "FP", "FLUOROURACIL": "FP"},
    regimen_names=(("FOLFIRINOX", frozenset({"FLUOROURACIL", "IRINOTECAN", "OXALIPLATIN"})),
                   ("GEM", frozenset({"GEMCITABINE"}))),
)


def episodes(*rows):
    return pd.DataFrame([("P", *r) for r in rows], columns=["PATIENT_ID", "REGIMEN", "START_DATE", "STOP_DATE"])


# --- lines ---------------------------------------------------------------------------------

@pytest.mark.parametrize("agents, name", [
    ("FLUOROURACIL_IRINOTECAN_LEUCOVORIN_OXALIPLATIN", "FOLFIRINOX"),
    ("GEMCITABINE_INVESTIGATIVE:Chemo", "TRIAL + GEM"),
    ("INVESTIGATIVE:Immuno", "TRIAL"),
    ("DOCETAXEL_GEMCITABINE", "DOCETAXEL+GEMCITABINE"),
    ("LEUCOVORIN", "UNKNOWN"),
])
def test_classify_regimen(agents, name):
    assert classify_regimen(agents, RULES) == name


def test_deescalation_and_maintenance_stay_in_line():
    ep = episodes(("FLUOROURACIL_IRINOTECAN_LEUCOVORIN_OXALIPLATIN", 0, 90),
                  ("FLUOROURACIL_IRINOTECAN_LEUCOVORIN", 91, 150),
                  ("CAPECITABINE", 160, 300))                      # equivalent to 5-FU
    L = build_lines_of_therapy(ep, RULES)
    assert len(L) == 1
    assert L.loc[0, ["REGIMEN", "LINE_START", "LINE_END", "N_EPISODES", "DEESCALATED"]].tolist() == \
        ["FOLFIRINOX", 0, 300, 3, True]


def test_agent_added_within_grace_joins_initial_regimen():
    ep = episodes(("FLUOROURACIL_IRINOTECAN_LEUCOVORIN", 0, 13), ("FLUOROURACIL_IRINOTECAN_LEUCOVORIN_OXALIPLATIN", 14, 90))
    L = build_lines_of_therapy(ep, RULES)
    assert L["REGIMEN"].tolist() == ["FOLFIRINOX"]


def test_new_agent_or_long_gap_starts_new_line():
    ep = episodes(("GEMCITABINE", 0, 100),
                  ("GEMCITABINE", 250, 300),                    # gap 149 > 90 days
                  ("FLUOROURACIL_IRINOTECAN_LEUCOVORIN_OXALIPLATIN", 310, 400))  # new agents
    L = build_lines_of_therapy(ep, RULES)
    assert L["LINE"].tolist() == [1, 2, 3]
    assert L["LINE_START"].tolist() == [0, 250, 310]


def test_lines_empty_input():
    assert build_lines_of_therapy(episodes(), RULES).empty


# --- follow-up and assessments -------------------------------------------------------------

def test_patient_followup():
    p = pd.DataFrame({"PATIENT_ID": ["A", "B", "C"], "OS_MONTHS": [12.0, 1.0, np.nan],
                      "OS_STATUS": ["1:DECEASED", "0:LIVING", "0:LIVING"]})
    fu = patient_followup(p)
    assert fu.values.tolist() == [["A", 365, 1], ["B", 30, 0]]


def test_collapse_assessments():
    im = pd.DataFrame({"PATIENT_ID": "P", "START_DATE": [1, 1, 2, 3, 4],
                       "PROCEDURE_TYPE": ["CT-TAB", "MR-H", "CT-TAB", "CT-TAB", "CT-TAB"],
                       "HAS_CANCER": ["N", "Y", "N", "Indeterminate", "N"],
                       "PROGRESSION": [None, "Y", "N", "Y", "Indeterminate"]})
    a = collapse_assessments(im)
    assert a[["DAY", "HAS_CANCER", "PROGRESSION", "PROCEDURES"]].values.tolist() == [
        [1, "Y", "Y", "CT-TAB,MR-H"], [2, "N", "N", "CT-TAB"], [4, "N", None, "CT-TAB"]]  # day 3 dropped


# --- line endpoints ------------------------------------------------------------------------

def _lines(*rows):
    return pd.DataFrame(rows, columns=["PATIENT_ID", "LINE", "LINE_START", "LINE_END"])


def _ass(*rows):
    return pd.DataFrame(rows, columns=["PATIENT_ID", "DAY", "PROGRESSION"])


def test_line_endpoints_progression_and_next_line():
    L = _lines(("P", 1, 0, 100), ("P", 2, 150, 300))
    A = _ass(("P", 20, "Y"), ("P", 90, "Y"), ("P", 200, "N"))     # day 20 within 30-day grace
    fu = pd.DataFrame({"PATIENT_ID": ["P"], "LAST_DAY": [400], "DEAD": [1]})
    E = line_endpoints(L, A, fu)
    first, second = E.iloc[0], E.iloc[1]
    assert (first.PFS_DAYS, first.PFS_EVENT, first.PFS_EVENT_TYPE) == (90, 1, "progression")
    assert (first.TTNT_DAYS, first.TTNT_EVENT, first.TTD_DAYS, first.TTD_EVENT) == (150, 1, 101, 1)
    # line 2: no progression, died 100 days after end of line (> 60) -> censored at last assessment
    assert (second.PFS_DAYS, second.PFS_EVENT, second.PFS_EVENT_TYPE) == (50, 0, "censored_last_assessment")
    assert (second.OS_DAYS, second.OS_EVENT, second.ENTRY) == (250, 1, 0)


def test_line_endpoints_death_ongoing_and_entry():
    L = _lines(("A", 1, -50, 100), ("B", 1, 0, 190))
    A = _ass(("A", 80, "N"))
    fu = pd.DataFrame({"PATIENT_ID": ["A", "B"], "LAST_DAY": [130, 200], "DEAD": [1, 0]})
    E = line_endpoints(L, A, fu).set_index("PATIENT_ID")
    assert (E.loc["A", "PFS_DAYS"], E.loc["A", "PFS_EVENT_TYPE"], E.loc["A", "ENTRY"]) == (180, "death", 50)
    assert (E.loc["B", "TTD_EVENT"], E.loc["B", "PFS_EVALUABLE"], E.loc["B", "PFS_EVENT_TYPE"]) == (0, False, "no_assessment")


def test_os_base_drops_unobservable():
    o = pd.DataFrame({"PATIENT_ID": ["A", "B"], "ORIGIN_DAY": [-100, -100]})
    fu = pd.DataFrame({"PATIENT_ID": ["A", "B"], "LAST_DAY": [50, -150], "DEAD": [1, 1]})
    b = os_base(o, fu)
    assert b[["PATIENT_ID", "OS_DAYS", "ENTRY"]].values.tolist() == [["A", 150, 100]]


# --- survival ------------------------------------------------------------------------------

def test_km_matches_lifelines_with_entry():
    from lifelines import KaplanMeierFitter
    rng = np.random.default_rng(1)
    t = rng.integers(1, 100, 200).astype(float)
    e = rng.integers(0, 2, 200)
    entry = np.minimum(rng.integers(0, 40, 200), t - 1)
    km = km_estimate(t, e, entry)
    ref = KaplanMeierFitter().fit(t, e, entry=entry)
    np.testing.assert_allclose(km["surv"], ref.survival_function_.iloc[:, 0].reindex(km["time"]), atol=1e-12)
    lower = ref.confidence_interval_.iloc[:, 0].reindex(km["time"]).to_numpy()
    defined = km["surv"].to_numpy() > 0
    np.testing.assert_allclose(km["lower"].to_numpy()[defined], lower[defined], atol=1e-6)


def test_counting_process():
    base = pd.DataFrame({"PATIENT_ID": ["A", "B"], "ORIGIN_DAY": [0, 0], "OS_DAYS": [100, 100],
                         "OS_EVENT": [1, 0], "ENTRY": [0, 20]})
    rt = pd.DataFrame({"PATIENT_ID": ["A", "B"], "RT_DAY": [40, 10]})   # B exposed before entry
    cp = to_counting_process(base, {"RT": rt})
    assert cp[["PATIENT_ID", "START", "STOP", "EVENT", "RT"]].values.tolist() == [
        ["A", 0, 40, 0, 0], ["A", 40, 100, 1, 1], ["B", 20, 100, 0, 1]]


def test_landmark():
    base = pd.DataFrame({"PATIENT_ID": ["A", "B", "C", "D"], "ORIGIN_DAY": 0, "OS_DAYS": [100, 100, 20, 100],
                         "OS_EVENT": 1, "ENTRY": [0, 0, 0, 90]})
    ex = pd.DataFrame({"PATIENT_ID": ["A", "B"], "X_DAY": [10, 60]})
    lm = landmark(base, ex, 30)
    assert lm[["PATIENT_ID", "EXPOSED", "LM_DAYS"]].values.tolist() == [["A", 1, 70], ["B", 0, 70]]


def test_benjamini_hochberg():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.2, np.nan])
    np.testing.assert_allclose(q[:4], [0.04, 0.04 * 4 / 3, 0.04 * 4 / 3, 0.2])
    assert np.isnan(q[4])


# --- covariates ----------------------------------------------------------------------------

def test_column_label_and_dx_parsing():
    assert column_label("CA19-9 (U/mL)") == "CA19_9"
    assert column_label("Lymph Nodes") == "LYMPH_NODES"
    p = parse_dx_description(pd.Series(["ADENOCARCINOMA, NOS | PANCREAS, HEAD (M8140/3 | C250)", "garbage"]))
    assert p.iloc[0].tolist() == ["ADENOCARCINOMA, NOS", "PANCREAS, HEAD", "8140/3", "C250"]
    assert p.iloc[1].isna().all()


def test_baseline_values_prefers_closest_then_before():
    v = pd.DataFrame({"PATIENT_ID": "P", "START_DATE": [-40, 95, 105, 200], "ECOG": [3, 1, 2, 0]})
    o = pd.DataFrame({"PATIENT_ID": ["P", "P"], "ORIGIN_DAY": [0, 100]})
    bl = baseline_values(v, o, "ECOG", window=(-30, 7))
    assert bl[["ORIGIN_DAY", "BL_ECOG", "BL_ECOG_OFFSET"]].values.tolist() == [[100, 1, -5]]


def test_line_history():
    L = pd.DataFrame({"PATIENT_ID": "P", "LINE": [1, 2], "LINE_START": [0, 200], "LINE_END": [100, 300],
                      "REGIMEN": ["FOLFIRINOX", "GEM"], "AGENTS_ALL": ["FLUOROURACIL_OXALIPLATIN", "GEMCITABINE"]})
    rt = pd.DataFrame({"PATIENT_ID": ["P"], "START_DATE": [150]})
    dx = pd.DataFrame({"PATIENT_ID": ["P"], "START_DATE": [-10]})
    h = line_history(L, dx, rt)
    assert h["PRIOR_OXALIPLATIN"].tolist() == [0, 1]
    assert h["PRIOR_GEMCITABINE"].tolist() == [0, 0]
    assert h["TFI_DAYS"].isna().tolist() == [True, False] and h["TFI_DAYS"].iloc[1] == 100
    assert h["PRIOR_RT"].tolist() == [0, 1]
    assert h["DAYS_DX_TO_LINE"].tolist() == [10, 210]


def test_marker_kinetics():
    L = pd.DataFrame({"PATIENT_ID": "P", "LINE_START": [0], "LINE_END": [120]})
    mk = pd.DataFrame({"PATIENT_ID": "P", "TEST": "CA19-9 (U/mL)", "START_DATE": [-5, 10, 30, 60],
                       "RESULT": [1000.0, 900.0, 600.0, 300.0]})
    k = marker_kinetics(L, mk, "CA19-9 (U/mL)", min_baseline=37).iloc[0]
    assert (k.CA19_9_NADIR, k.CA19_9_BEST_PCT, k.CA19_9_N_ONTX) == (300.0, -70.0, 2)  # day 10 < min_day 14
    assert (k.CA19_9_PCT_56D, k.CA19_9_EVALUABLE, k.CA19_9_RESPONSE) == (-70.0, 1, 1.0)


def test_genomic_features_panel_mask(tmp_path):
    pd.DataFrame({"Hugo_Symbol": ["KRAS", "BRCA2", "TP53"], "Tumor_Sample_Barcode": ["S1", "S1", "S2"],
                  "Variant_Classification": ["Missense_Mutation", "Nonsense_Mutation", "Silent"],
                  "HGVSp_Short": ["p.G12D", "p.R3052*", "p.P72P"]}).to_csv(tmp_path / "data_mutations.txt", sep="\t", index=False)
    pd.DataFrame({"Hugo_Symbol": ["KRAS", "BRCA2"], "S1": [0, 0], "S2": [2, -2]}).to_csv(
        tmp_path / "data_cna.txt", sep="\t", index=False)
    pd.DataFrame({"Sample_Id": ["S2"], "Site1_Hugo_Symbol": ["TP53"], "Site2_Hugo_Symbol": ["X"]}).to_csv(
        tmp_path / "data_sv.txt", sep="\t", index=False)
    pd.DataFrame({"SAMPLE_ID": ["S1", "S2"], "mutations": ["NEW", "OLD"]}).to_csv(
        tmp_path / "data_gene_panel_matrix.txt", sep="\t", index=False)
    study = cu.ChordStudy(tmp_path)

    g = genomic_features(study, ["S1", "S2"], ["KRAS", "TP53"], {"HR": ["BRCA2"]}, protein_changes=["KRAS"]).set_index("SAMPLE_ID")
    assert g["KRAS_ALT"].tolist() == [1, 1]            # S1 mutation, S2 amplification
    assert g["TP53_ALT"].tolist() == [0, 1]            # silent not counted; S2 fusion
    assert g["TP53_HOMDEL"].isna().all() if "TP53_HOMDEL" in g else True   # no CNA row -> missing
    assert g["HR_ALT"].tolist() == [1, 1] and g["KRAS_PROTEIN"].tolist()[0] == "p.G12D"
    assert g["N_NONSYN"].tolist() == [2, 0]

    masked = genomic_features(study, ["S1", "S2"], ["BRCA2"], panel_genes={"NEW": ["BRCA2"], "OLD": []}).set_index("SAMPLE_ID")
    assert masked["BRCA2_ALT"].iloc[0] == 1 and np.isnan(masked["BRCA2_ALT"].iloc[1])
