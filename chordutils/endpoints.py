"""Follow-up, disease assessments and real-world endpoints (OS, TTD, TTNT, rwPFS).

Time axis
---------
All days are on the MSK-CHORD timeline axis (day 0 = first sequenced sample). OS_MONTHS in
``data_clinical_patient.txt`` is measured from the same day 0 in months of 365/12 days:
``OS_MONTHS * 12 / 365`` is an integer number of days for every patient (maximal deviation
< 0.004 days), and with this conversion no timeline event of any patient lies after the day
of death / last contact (checked 2026-09-29 on the full study).

Left truncation
---------------
Patients are only in MSK-CHORD because a tumour sample was sequenced. Whoever has a time
origin (diagnosis, start of a line, ...) BEFORE day 0 had to survive until day 0 to be
included. Endpoint tables therefore carry ENTRY = days from origin to day 0 (0 if the origin
is after day 0), to be used as delayed entry for OS (``survival.km_estimate(..., entry=)``,
lifelines ``entry_col``, R ``Surv(entry, time, event)``). For TTD/TTNT/rwPFS the events can
themselves occur before sequencing, so delayed entry is not a valid correction; restrict to an
inception cohort (e.g. ``ENTRY <= 30``) and report all lines as sensitivity analysis.

Adapted from an earlier PDAC analysis module (chordendpoints).
"""

import numpy as np
import pandas as pd

ID = "PATIENT_ID"

DAYS_PER_MONTH = 365 / 12


def patient_followup(patients: pd.DataFrame) -> pd.DataFrame:
    """Day of death or last contact on the timeline axis.

    Parameters
    ----------
    patients : pd.DataFrame
        Patient table with OS_MONTHS and OS_STATUS (1/0 or '1:DECEASED'/'0:LIVING'),
        e.g. ``Cohort.patients``.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, LAST_DAY (= round(OS_MONTHS * 365/12)), DEAD (1/0). Patients without
        OS_MONTHS are dropped.
    """

    d = patients[[ID, "OS_MONTHS", "OS_STATUS"]].dropna(subset=["OS_MONTHS"])
    status = d["OS_STATUS"]
    if not pd.api.types.is_numeric_dtype(status):
        status = status.astype(str).str.split(":").str[0]
    return pd.DataFrame({ID: d[ID], "LAST_DAY": np.round(d["OS_MONTHS"] * DAYS_PER_MONTH).astype(int),
                         "DEAD": status.astype(int)}).reset_index(drop=True)


def collapse_assessments(imaging: pd.DataFrame) -> pd.DataFrame:
    """Combine all imaging reports of one day into one disease assessment.

    MSK-CHORD has no report identifier, and several reports can exist per day (e.g. CT
    chest/abdomen/pelvis + MRI brain, or two reports of the same modality). Joining the NLP
    tables per report is therefore ambiguous (joining them yields duplicated and mismatched rows); a per-day assessment
    is not.

    Rules per patient and day ('Indeterminate' is treated as missing):

    - HAS_CANCER = 'Y' if any report says 'Y', 'N' if at least one says 'N' and none 'Y'.
      Days without any determinate HAS_CANCER call are dropped.
    - PROGRESSION = 'Y' if any report says 'Y', 'N' if at least one says 'N' and none 'Y',
      else missing.

    Parameters
    ----------
    imaging : pd.DataFrame
        ``tables.imaging_assessments`` (db['STAGING']) with PATIENT_ID, START_DATE,
        PROCEDURE_TYPE, HAS_CANCER, PROGRESSION.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, DAY, HAS_CANCER, PROGRESSION, PROCEDURES (distinct procedures of the day,
        joined by ',').
    """

    d = imaging.replace({"HAS_CANCER": {"Indeterminate": np.nan}, "PROGRESSION": {"Indeterminate": np.nan}})
    flags = pd.DataFrame({ID: d[ID], "DAY": d["START_DATE"],
                          "hc_y": d["HAS_CANCER"].eq("Y"), "hc_n": d["HAS_CANCER"].eq("N"),
                          "pr_y": d["PROGRESSION"].eq("Y"), "pr_n": d["PROGRESSION"].eq("N")})
    g = flags.groupby([ID, "DAY"])[["hc_y", "hc_n", "pr_y", "pr_n"]].any()
    g["PROCEDURES"] = d.groupby([ID, "START_DATE"])["PROCEDURE_TYPE"].agg(
        lambda s: ",".join(sorted(set(s.dropna())))).to_numpy()
    g = g.reset_index()

    g["HAS_CANCER"] = np.select([g["hc_y"], g["hc_n"]], ["Y", "N"], default=None)
    g["PROGRESSION"] = np.select([g["pr_y"], g["pr_n"]], ["Y", "N"], default=None)
    out = g[g["HAS_CANCER"].notna()][[ID, "DAY", "HAS_CANCER", "PROGRESSION", "PROCEDURES"]]
    return out.sort_values([ID, "DAY"], ignore_index=True)


def line_endpoints(lines: pd.DataFrame,
                   assessments: pd.DataFrame,
                   followup: pd.DataFrame,
                   ongoing_window: int = 30,
                   progression_grace: int = 30,
                   death_window: int = 60,
                   censor_at_next_line: bool = True) -> pd.DataFrame:
    """TTD, TTNT, rwPFS and OS for every line of therapy (time origin = line start).

    Definitions (all times in days from LINE_START):

    TTD (time to treatment discontinuation)
        LINE_END - LINE_START + 1. Censored (TTD_EVENT = 0) if the patient is alive, has no
        next line and LINE_END is within ``ongoing_window`` days of last contact (therapy
        possibly ongoing at data cut-off).
    TTNT (time to next treatment or death)
        Start of the next line (event) or death (event); otherwise censored at last contact.
    rwPFS (real-world progression-free survival)
        First assessment with PROGRESSION = 'Y' at least ``progression_grace`` days after the
        line start (earlier scans reflect the baseline) and not after the start of the next
        line -> event 'progression'. Otherwise death -> event 'death' if the patient has no
        next line and died within ``death_window`` days after the later of the last
        assessment and LINE_END. Otherwise censored at the start of the next line
        (``censor_at_next_line``, switch without documented progression) or at the last
        assessment. Lines without any assessment after the grace period have
        PFS_EVALUABLE = False (PFS_DAYS = 0, 'no_assessment').
    OS
        Death or last contact.

    Parameters
    ----------
    lines : pd.DataFrame
        ``lines.build_lines_of_therapy`` output (PATIENT_ID, LINE_START, LINE_END, ...).
    assessments : pd.DataFrame
        ``collapse_assessments`` output (PATIENT_ID, DAY, PROGRESSION).
    followup : pd.DataFrame
        ``patient_followup`` output.

    Returns
    -------
    pd.DataFrame
        ``lines`` + LAST_DAY, DEAD, NEXT_LINE_START, TTD_DAYS/_EVENT, TTNT_DAYS/_EVENT,
        PFS_DAYS/_EVENT/_EVENT_TYPE, PFS_EVALUABLE, OS_DAYS/_EVENT and ENTRY (see module
        docstring on left truncation).
    """

    d = lines.sort_values([ID, "LINE_START"]).merge(followup, on=ID, how="left")
    d["NEXT_LINE_START"] = d.groupby(ID)["LINE_START"].shift(-1)

    by_patient = {k: (v["DAY"].to_numpy(), v["PROGRESSION"].to_numpy()) for k, v in assessments.groupby(ID)}
    empty = (np.array([]), np.array([]))
    res = {k: [] for k in ["TTD_DAYS", "TTD_EVENT", "TTNT_DAYS", "TTNT_EVENT", "PFS_DAYS", "PFS_EVENT",
                           "PFS_EVENT_TYPE", "PFS_EVALUABLE"]}

    def add(**kw):
        for k, v in kw.items():
            res[k].append(v)

    for r in d.itertuples(index=False):

        start, end, last_day, dead, nxt = r.LINE_START, r.LINE_END, r.LAST_DAY, r.DEAD, r.NEXT_LINE_START
        has_next = pd.notnull(nxt)

        ongoing = dead == 0 and last_day - end <= ongoing_window and not has_next
        add(TTD_DAYS=end - start + 1, TTD_EVENT=0 if ongoing else 1)

        if has_next:
            add(TTNT_DAYS=nxt - start, TTNT_EVENT=1)
        else:
            add(TTNT_DAYS=last_day - start, TTNT_EVENT=int(dead == 1))

        day, prog = by_patient.get(getattr(r, ID), empty)
        upper = nxt if has_next else last_day
        window = (day >= start + progression_grace) & (day <= upper)
        progressed = window & (prog == "Y")
        add(PFS_EVALUABLE=bool(window.any()))

        if progressed.any():
            add(PFS_DAYS=day[progressed][0] - start, PFS_EVENT=1, PFS_EVENT_TYPE="progression")
            continue

        last_assessment = day[window].max() if window.any() else np.nan
        reference = np.nanmax([last_assessment, end])

        if dead == 1 and not has_next and last_day - reference <= death_window:
            add(PFS_DAYS=last_day - start, PFS_EVENT=1, PFS_EVENT_TYPE="death")
        elif has_next and censor_at_next_line:
            add(PFS_DAYS=nxt - start, PFS_EVENT=0, PFS_EVENT_TYPE="censored_next_line")
        elif pd.notnull(last_assessment):
            add(PFS_DAYS=last_assessment - start, PFS_EVENT=0, PFS_EVENT_TYPE="censored_last_assessment")
        else:
            add(PFS_DAYS=0, PFS_EVENT=0, PFS_EVENT_TYPE="no_assessment")

    for k, v in res.items():
        d[k] = v
    d["OS_DAYS"] = d["LAST_DAY"] - d["LINE_START"]
    d["OS_EVENT"] = d["DEAD"]
    d["ENTRY"] = np.clip(-d["LINE_START"], 0, None)
    return d.reset_index(drop=True)


def os_base(origins: pd.DataFrame, followup: pd.DataFrame, origin_col: str = "ORIGIN_DAY") -> pd.DataFrame:
    """Overall survival from an arbitrary time origin (diagnosis, line start, resection, ...).

    Parameters
    ----------
    origins : pd.DataFrame
        One row per patient with PATIENT_ID and ``origin_col`` (timeline day); further
        columns are kept.
    followup : pd.DataFrame
        ``patient_followup`` output.

    Returns
    -------
    pd.DataFrame
        ``origins`` + LAST_DAY, DEAD, OS_DAYS, OS_EVENT, ENTRY. Patients whose follow-up ends
        before their entry time cannot be observed and are dropped.
    """

    d = origins.merge(followup, on=ID)
    d["OS_DAYS"] = d["LAST_DAY"] - d[origin_col]
    d["OS_EVENT"] = d["DEAD"]
    d["ENTRY"] = np.clip(-d[origin_col], 0, None)
    observable = (d["OS_DAYS"] > d["ENTRY"]) | ((d["OS_DAYS"] == 0) & (d["ENTRY"] == 0))
    return d[observable].reset_index(drop=True)
