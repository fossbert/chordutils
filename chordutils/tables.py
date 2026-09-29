"""Analysis tables of a cohort: therapy episodes, surgery, radiation, imaging, markers, ECOG.

Each function takes a ``Cohort`` (see ``cohort.select_cohort``) and the ``ChordStudy`` and
returns one tidy table restricted to the cohort. Every filter step is written to
``cohort.log``. ``build_database`` assembles all tables into the dictionary format of
the PDAC analysis database (dict of DataFrames).

Unless stated otherwise, event tables get a column TIME_FROM_DIAGNOSIS (event day minus day of
diagnosis) and are restricted to ``config.window(<timeline>)``.
"""

from typing import Dict

import numpy as np
import pandas as pd

from .cohort import ID, Cohort
from .io import ChordStudy
from .therapy import pivot_therapies

# Imaging regions of the cancer-presence timeline and their one-letter codes, following the
# German radiology convention: T(horax), A(bdomen), B(ecken = pelvis), H(ead), O(ther),
# e.g. 'CT-TAB' = CT of chest, abdomen and pelvis.
REGION_CODES = {"CHEST": "T", "ABDOMEN": "A", "PELVIS": "B", "HEAD": "H", "OTHER": "O"}


def _finish(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Select columns, order rows by patient and day, fresh index."""

    return d[list(cols)].sort_values([ID, "START_DATE"], kind="stable").reset_index(drop=True)


def treatment_episodes(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """Regimen episodes of systemic therapy (one row per episode).

    Steps (rules from ``config.treatment``, each logged):

    1. cohort patients; SUBTYPE in ``subtypes``; AGENT in ``agents`` (if given)
    2. remove raw agent rows with ``STOP_DATE - START_DATE < min_row_span[agent]``
    3. ``pivot_therapies`` per patient (``min_size = pivot_min_size``)
    4. remove episodes shorter than ``min_episode_days`` and episodes of a regimen in
       ``min_regimen_days`` shorter than the given number of days
    5. time from diagnosis of the episode start; keep ``config.window('treatment')``

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, REGIMEN (agents joined by '_'), START_DATE, STOP_DATE (both included),
        DAYS, TIME_FROM_DIAGNOSIS.
    """

    rules, log, t = cohort.config.treatment, cohort.log, "treatment"

    tx = cohort.restrict(study.timeline("treatment"), table=t)
    tx = log.filter(tx, tx["SUBTYPE"].isin(rules.subtypes), t, "treatment subtypes", ", ".join(rules.subtypes))
    if rules.agents is not None:
        tx = log.filter(tx, tx["AGENT"].isin(rules.agents), t, "agents", f"{len(rules.agents)} agents of interest")
    for agent, span in rules.min_row_span.items():
        short = (tx["AGENT"] == agent) & ((tx["STOP_DATE"] - tx["START_DATE"]) < span)
        tx = log.filter(tx, ~short, t, f"short {agent} records", f"STOP_DATE - START_DATE < {span} days")

    cols = ["AGENT", "START_DATE", "STOP_DATE", "RX_INVESTIGATIVE", "SUBTYPE"]
    ep = (tx.groupby(ID)[cols].apply(pivot_therapies, min_size=rules.pivot_min_size)
            .reset_index(level=0).reset_index(drop=True))
    ep = ep.rename(columns={"agent": "REGIMEN", "start": "START_DATE", "stop": "STOP_DATE", "days": "DAYS"})
    log.record(tx, ep, t, "regimen episodes", "pivot_therapies: agent rows -> non-overlapping episodes",
               detail=f"min_size={rules.pivot_min_size}")

    ep = log.filter(ep, ep["DAYS"] >= rules.min_episode_days, t, "short episodes",
                    f"fewer than {rules.min_episode_days} treatment days")
    for regimen, days in rules.min_regimen_days.items():
        short = (ep["REGIMEN"] == regimen) & (ep["DAYS"] < days)
        ep = log.filter(ep, ~short, t, f"short {regimen} episodes", f"{regimen} alone for fewer than {days} days")

    ep = cohort.align(ep, t, restrict=False)
    return _finish(ep, [ID, "REGIMEN", "START_DATE", "STOP_DATE", "DAYS", "TIME_FROM_DIAGNOSIS"])


def surgeries(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """Surgical events with the sample taken, if any.

    The surgery timeline has no procedure type, only SUBTYPE 'PROCEDURE' or 'SAMPLE'. Samples
    are linked through ``data_timeline_specimen_surgery.txt`` (same patient and day), and the
    sample's type (Primary/Metastasis) and metastatic site are added from ``cohort.samples``.

    If one surgery is linked to several samples (e.g. pancreas and colon resected in the same
    operation) and at least one of them belongs to the cohort, only the cohort samples are
    kept. Links to other samples of the patient (e.g. a second, non-cohort tumour) are kept
    otherwise, with missing SAMPLE_TYPE.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START_DATE, SUBTYPE, TIME_FROM_DIAGNOSIS, SAMPLE_ID, SEQ_DATE, SAMPLE_TYPE,
        METASTATIC_SITE.
    """

    log, t = cohort.log, "surgery"

    sx = cohort.align(study.timeline("surgery"), t)[[ID, "START_DATE", "SUBTYPE", "TIME_FROM_DIAGNOSIS"]]

    spec = study.timeline("specimen_surgery")[[ID, "START_DATE", "SAMPLE_ID", "SEQ_DATE"]]
    linked = sx.reset_index(drop=True).rename_axis("_row").reset_index().merge(spec, on=[ID, "START_DATE"], how="left")
    log.record(sx, linked, t, "link samples", "specimen_surgery on patient and day (left join)")

    in_cohort = linked["SAMPLE_ID"].isin(cohort.samples["SAMPLE_ID"])
    has_cohort_sample = in_cohort.groupby(linked["_row"]).transform("any")
    multi = linked["_row"].duplicated(keep=False)
    linked = log.filter(linked, ~(multi & has_cohort_sample & ~in_cohort), t, "one sample per surgery",
                        "surgery linked to several samples: keep the cohort sample")

    linked = linked.merge(cohort.samples[[ID, "SAMPLE_ID", "SAMPLE_TYPE", "METASTATIC_SITE"]],
                          on=[ID, "SAMPLE_ID"], how="left")
    return _finish(linked, [ID, "START_DATE", "SUBTYPE", "TIME_FROM_DIAGNOSIS", "SAMPLE_ID", "SEQ_DATE",
                            "SAMPLE_TYPE", "METASTATIC_SITE"])


def radiation(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """Start days of radiation courses (the timeline has no target region, dose or intent).

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START_DATE, TIME_FROM_DIAGNOSIS.
    """

    rt = cohort.align(study.timeline("radiation"), "radiation")
    return _finish(rt, [ID, "START_DATE", "TIME_FROM_DIAGNOSIS"])


def performance_status(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """ECOG performance status records.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START_DATE, ECOG, TIME_FROM_DIAGNOSIS.
    """

    ps = cohort.align(study.timeline("performance_status"), "performance_status")
    return _finish(ps, [ID, "START_DATE", "ECOG", "TIME_FROM_DIAGNOSIS"])


def tumor_markers(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """Tumour marker values of the tests in ``config.tumor_markers``.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START_DATE, TEST, RESULT, TIME_FROM_DIAGNOSIS; ordered by patient and day.
    """

    log, t, tests = cohort.log, "tumor_markers", list(cohort.config.tumor_markers)

    tm = cohort.align(study.timeline("tumor_markers"), t)
    tm = log.filter(tm, tm["TEST"].isin(tests), t, "marker tests", ", ".join(tests))
    return _finish(tm, [ID, "START_DATE", "TEST", "RESULT", "TIME_FROM_DIAGNOSIS"])


def region_code(regions: pd.DataFrame, codes: Dict[str, str] = REGION_CODES) -> pd.Series:
    """One-letter codes of the imaged regions per row, e.g. CHEST+ABDOMEN+PELVIS -> 'TAB'."""

    flags = regions[list(codes)].astype(bool).to_numpy()
    letters = np.array(list(codes.values()))
    return pd.Series(["".join(letters[row]) for row in flags], index=regions.index)


def imaging_assessments(cohort: Cohort, study: ChordStudy) -> pd.DataFrame:
    """Imaging reports with NLP-derived cancer presence, progression and tumour sites.

    Combines three NLP timelines on patient, day and procedure (CT, MR, PET, ...):

    - ``cancer_presence`` (HAS_CANCER Y/N; 'Indeterminate' removed) is the base table,
    - ``progression`` (PROGRESSION Y/N; 'Indeterminate' removed) is added by left join, so
      progression calls without a cancer-presence call on the same report are not kept,
    - ``tumor_sites`` (one row per site) are joined by ',' per report and added as TUMOR_SITE.

    PROCEDURE_TYPE is extended by the imaged regions (see ``REGION_CODES``), e.g. 'CT-TAB'.
    No time window is applied unless ``config.windows`` contains 'imaging'; imaging before the
    diagnosis is often the diagnostic work-up. TIME_FROM_DIAGNOSIS is added only in that case.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, START_DATE, PROCEDURE_TYPE, HAS_CANCER, PROGRESSION, TUMOR_SITE
        (+ TIME_FROM_DIAGNOSIS if a window is configured).
    """

    log, t, keys = cohort.log, "imaging", [ID, "START_DATE", "PROCEDURE_TYPE"]

    cp = cohort.restrict(study.timeline("cancer_presence"), table=t)
    cp = log.filter(cp, cp["HAS_CANCER"] != "Indeterminate", t, "determinate cancer presence",
                    "HAS_CANCER 'Indeterminate' removed")
    cp = cp[keys + list(REGION_CODES) + ["HAS_CANCER"]]

    pr = cohort.restrict(study.timeline("progression"), table="progression")
    pr = log.filter(pr, pr["PROGRESSION"] != "Indeterminate", "progression", "determinate progression",
                    "PROGRESSION 'Indeterminate' removed")
    merged = cp.merge(pr[keys + ["PROGRESSION"]], on=keys, how="left")
    log.record(cp, merged, t, "add progression", "left join on patient, day, procedure")

    ts = cohort.restrict(study.timeline("tumor_sites"), table="tumor_sites")
    ts = (ts.groupby([ID, "START_DATE", "SOURCE_SPECIFIC"])["TUMOR_SITE"].agg(",".join).reset_index()
            .rename(columns={"SOURCE_SPECIFIC": "PROCEDURE_TYPE"}))
    before = merged
    merged = merged.merge(ts, on=keys, how="left")
    log.record(before, merged, t, "add tumour sites", "sites per report joined by ','")

    merged["PROCEDURE_TYPE"] = merged["PROCEDURE_TYPE"] + "-" + region_code(merged)
    cols = keys + ["HAS_CANCER", "PROGRESSION", "TUMOR_SITE"]

    if "imaging" in cohort.config.windows:
        merged = cohort.align(merged, t, restrict=False)
        cols = cols + ["TIME_FROM_DIAGNOSIS"]
    return merged[cols].reset_index(drop=True)


def build_database(cohort: Cohort, study: ChordStudy) -> Dict[str, pd.DataFrame]:
    """All tables of a cohort as a dict of DataFrames (the PDAC database format).

    Keys: SAMPLE, CLINICAL, DIAGNOSIS, TREATMENT, SURGERY, RADIATION, STAGING, TUMORMARKER,
    PERFORMANCE. The derivation of every table is recorded in ``cohort.log``.
    """

    return {
        "SAMPLE": cohort.samples.reset_index(drop=True),
        "CLINICAL": cohort.patients.reset_index(drop=True),
        "DIAGNOSIS": cohort.diagnosis.reset_index(drop=True),
        "TREATMENT": treatment_episodes(cohort, study),
        "SURGERY": surgeries(cohort, study),
        "RADIATION": radiation(cohort, study),
        "STAGING": imaging_assessments(cohort, study),
        "TUMORMARKER": tumor_markers(cohort, study),
        "PERFORMANCE": performance_status(cohort, study),
    }
