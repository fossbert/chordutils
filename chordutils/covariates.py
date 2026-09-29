"""Patient-, tumour-, genomic- and line-level covariates.

Everything here can be merged onto the line table from ``endpoints.line_endpoints`` (one row
per line of therapy); ``build_line_table`` does this in one call.

Caveats
-------
- On-treatment variables (marker response, nadir) are only known after some time on
  therapy. Correlating them with TTD/rwPFS/OS from the line start introduces guarantee-time
  bias; use a landmark (``survival.landmark``).
- NLP tumour sites include the organ of the primary tumour; see
  ``EntityConfig.primary_site_categories``.
- Genomic calls are somatic (MSK-IMPACT is tumour/normal matched). The gene content of the
  panel versions is not part of the download; see ``genomic_features`` (panel coverage).

Adapted from an earlier PDAC analysis module (chordcovariates); PDAC-specific parts (primary location in
the pancreas, KRAS allele classes, CA19-9 non-secretors) were left out.
"""

import re
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import EntityConfig

ID = "PATIENT_ID"

# Variant classifications that change the protein sequence (MAF nomenclature).
NONSYNONYMOUS = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
                 "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation"}


def column_label(text: str) -> str:
    """Column-safe label: 'CA19-9 (U/mL)' -> 'CA19_9', 'Lymph Nodes' -> 'LYMPH_NODES'."""

    text = re.sub(r"\s*\(.*\)\s*$", "", text)
    return re.sub(r"[^A-Z0-9]+", "_", text.upper()).strip("_")


# --- patient level -------------------------------------------------------------------------

def parse_dx_description(dx_description: pd.Series) -> pd.DataFrame:
    """Split DX_DESCRIPTION into its parts.

    'ADENOCARCINOMA, NOS | PANCREAS, HEAD (M8140/3 | C250)' ->
    HISTOLOGY_DX='ADENOCARCINOMA, NOS', SITE_DX='PANCREAS, HEAD', MORPHOLOGY_CODE='8140/3',
    TOPOGRAPHY_CODE='C250'. Entries that do not follow the pattern give missing values.
    """

    pattern = r"^\s*(?P<HISTOLOGY_DX>[^|]*?)\s*\|\s*(?P<SITE_DX>.*?)\s*\(M(?P<MORPHOLOGY_CODE>[^|]+?)\s*\|\s*(?P<TOPOGRAPHY_CODE>C\d+)\)\s*$"
    return dx_description.str.extract(pattern)


def patient_covariates(cohort, study=None) -> pd.DataFrame:
    """Static patient and tumour covariates (one row per patient).

    Combines ``cohort.patients`` (sex, race, ethnicity, smoking prediction, ...), the entity
    diagnosis (day, stage, registry groups, parsed histology/site) and the first sequenced
    sample of the cohort (sample type, metastatic site, panel, purity, MSI, sequencing day).
    With ``study``, SAMPLE_ACQ_DAY (day the tissue was obtained, from
    data_timeline_specimen_surgery.txt) is added; it can lie long before SEQ_DATE, the day of
    sequencing.

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, SEX, ... (patient columns as available), DX_DAY, STAGE_DX, CLINICAL_GROUP,
        PATH_GROUP, SUMMARY, HISTOLOGY_DX, SITE_DX, MORPHOLOGY_CODE, TOPOGRAPHY_CODE,
        SAMPLE_ID, SAMPLE_TYPE, ..., SEQ_DATE (+ SAMPLE_ACQ_DAY).
    """

    patient_cols = ["GENDER", "RACE", "ETHNICITY", "SMOKING_PREDICTIONS_3_CLASSES", "PRIOR_MED_TO_MSK",
                    "NUM_ICDO_DX", "HISTORY_OF_PDL1", "HR", "HER2"]
    p = cohort.patients
    out = p[[ID] + [c for c in patient_cols if c in p.columns]].rename(
        columns={"GENDER": "SEX", "SMOKING_PREDICTIONS_3_CLASSES": "SMOKING"})

    dx = cohort.diagnosis
    dx = pd.concat([dx, parse_dx_description(dx["DX_DESCRIPTION"])], axis=1)
    dx_cols = ["START_DATE", "STAGE_CDM_DERIVED", "CLINICAL_GROUP", "PATH_GROUP", "SUMMARY", "HISTOLOGY_DX",
               "SITE_DX", "MORPHOLOGY_CODE", "TOPOGRAPHY_CODE"]
    dx = dx[[ID] + [c for c in dx_cols if c in dx.columns]].rename(
        columns={"START_DATE": "DX_DAY", "STAGE_CDM_DERIVED": "STAGE_DX"})
    out = out.merge(dx, on=ID, how="left")

    sample_cols = ["SAMPLE_ID", "SAMPLE_TYPE", "METASTATIC_SITE", "PRIMARY_SITE", "CANCER_TYPE_DETAILED",
                   "GENE_PANEL", "TUMOR_PURITY", "MSI_TYPE", "MSI_SCORE", "PDL1_POSITIVE", "SEQ_DATE"]
    s = cohort.samples.sort_values([ID, "SEQ_DATE"], kind="stable").drop_duplicates(ID)
    s = s[[ID] + [c for c in sample_cols if c in s.columns]]
    out = out.merge(s, on=ID, how="left")

    if study is not None:
        acq = study.timeline("specimen_surgery")[["SAMPLE_ID", "START_DATE"]].drop_duplicates("SAMPLE_ID")
        out = out.merge(acq.rename(columns={"START_DATE": "SAMPLE_ACQ_DAY"}), on="SAMPLE_ID", how="left")
    return out


# --- genomics ------------------------------------------------------------------------------

def genomic_features(study, sample_ids: Sequence[str], genes: Sequence[str] = (),
                     gene_groups: Optional[Dict[str, Sequence[str]]] = None,
                     protein_changes: Sequence[str] = (),
                     panel_genes: Optional[Dict[str, Sequence[str]]] = None,
                     panel_sizes_mb: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Alteration calls per sample from the MSK-CHORD genomic files.

    Per gene (``genes`` plus all genes of ``gene_groups``):

    - <GENE>_MUT: nonsynonymous mutation (``NONSYNONYMOUS``; somatic calls, VUS included, no
      OncoKB filter)
    - <GENE>_HOMDEL / <GENE>_AMP: data_cna.txt value -2 / 2 (other values such as -1.5 are
      not counted). Missing for genes without copy-number values in data_cna.txt (161 of its
      702 genes have no value in any sample).
    - <GENE>_FUSION: gene is a partner in data_sv.txt
    - <GENE>_ALT: any of the above

    Per group: <GROUP>_N (number of altered genes) and <GROUP>_ALT (1 if any gene is altered,
    0 if no gene is altered and all were profiled, else missing).

    Panel coverage
    --------------
    The MSK-IMPACT panel versions differ in their genes (IMPACT341 < 410 < 468 < 505, see
    GENE_PANEL / data_gene_panel_matrix.txt). The MSK-CHORD download does not contain the gene
    lists of the panels, and data_cna.txt reports 0 (not missing) for genes that were not on a
    sample's panel. Without ``panel_genes`` a gene that was not sequenced therefore appears as
    NOT altered. Pass ``panel_genes`` (from the cBioPortal gene panel files, or derived from an
    alteration export with ``cbioportal.panel_genes_from_export``) to set all calls of genes
    outside a sample's panel to missing; otherwise consider GENE_PANEL in the analysis (e.g. a
    sensitivity analysis without IMPACT341 samples) for genes that were added in later panel
    versions. Genes that appear in none of the ``panel_genes`` lists are left unmasked (their
    panel coverage is unknown, e.g. genes outside the query of an export).

    Parameters
    ----------
    study : ChordStudy
    sample_ids : sequence of str
        Samples to annotate (e.g. ``cohort.samples.SAMPLE_ID``).
    genes, gene_groups : see ``EntityConfig``
    protein_changes : sequence of str
        Genes for which the protein changes (HGVSp_Short, e.g. 'p.G12D') are listed as
        <GENE>_PROTEIN (joined by ','), e.g. to classify KRAS alleles.
    panel_genes : dict of {panel: genes}, optional
        Genes on each panel (keys as in data_gene_panel_matrix.txt, column 'mutations'). Only
        genes listed for at least one panel are masked.
    panel_sizes_mb : dict of {panel: Mb}, optional
        Coding size per panel; if given, TMB = N_NONSYN / size is added. No defaults are
        provided on purpose: use the sizes of the panel version you cite.

    Returns
    -------
    pd.DataFrame
        One row per sample: SAMPLE_ID, gene and group columns, N_NONSYN (+ TMB).
    """

    gene_groups = gene_groups or {}
    sids = pd.Index(pd.unique(pd.Series(list(sample_ids))), name="SAMPLE_ID")
    all_genes = list(dict.fromkeys([*genes, *(g for gl in gene_groups.values() for g in gl)]))

    mut = study.read("data_mutations.txt", usecols=["Hugo_Symbol", "Tumor_Sample_Barcode",
                                                     "Variant_Classification", "HGVSp_Short"])
    mut = mut[mut["Tumor_Sample_Barcode"].isin(sids) & mut["Variant_Classification"].isin(NONSYNONYMOUS)]
    cna = study.read("data_cna.txt").set_index("Hugo_Symbol").reindex(columns=sids)
    sv = study.read("data_sv.txt", usecols=["Sample_Id", "Site1_Hugo_Symbol", "Site2_Hugo_Symbol"])
    sv = sv[sv["Sample_Id"].isin(sids)]
    panel = study.read("data_gene_panel_matrix.txt").set_index("SAMPLE_ID")["mutations"].reindex(sids)

    mutated = set(zip(mut["Tumor_Sample_Barcode"], mut["Hugo_Symbol"]))
    fused = set(zip(sv["Sample_Id"], sv["Site1_Hugo_Symbol"])) | set(zip(sv["Sample_Id"], sv["Site2_Hugo_Symbol"]))

    known_panel_genes = {g for gl in (panel_genes or {}).values() for g in gl}
    cols = {}
    for g in all_genes:
        cn = cna.loc[g] if g in cna.index else pd.Series(np.nan, index=sids)
        calls = {"MUT": pd.Series([float((s, g) in mutated) for s in sids], index=sids),
                 "HOMDEL": (cn == -2).astype(float).where(cn.notna()),
                 "AMP": (cn == 2).astype(float).where(cn.notna()),
                 "FUSION": pd.Series([float((s, g) in fused) for s in sids], index=sids)}
        if panel_genes is not None and g in known_panel_genes:
            on_panel = panel.map(lambda p, g=g: g in panel_genes.get(p, ())).astype(bool)
            calls = {k: v.where(on_panel) for k, v in calls.items()}
        for kind, call in calls.items():
            cols[f"{g}_{kind}"] = call
        # ALT: 1 if any call is positive, 0 if the gene was sequenced (MUT is only missing when
        # the gene is off-panel) and nothing was found; missing CNA alone does not make ALT missing
        any_pos = pd.concat(calls.values(), axis=1).eq(1).any(axis=1)
        cols[f"{g}_ALT"] = pd.Series(np.where(any_pos, 1.0, np.where(calls["MUT"].notna(), 0.0, np.nan)), index=sids)

    out = pd.DataFrame(cols, index=sids)

    for grp, gl in gene_groups.items():
        alt = out[[f"{g}_ALT" for g in gl]]
        out[f"{grp}_N"] = alt.sum(axis=1)
        out[f"{grp}_ALT"] = np.where(alt.eq(1).any(axis=1), 1.0, np.where(alt.notna().all(axis=1), 0.0, np.nan))

    for g in protein_changes:
        pc = mut[mut["Hugo_Symbol"] == g].groupby("Tumor_Sample_Barcode")["HGVSp_Short"].agg(
            lambda s: ",".join(sorted(set(s.dropna()))))
        out[f"{g}_PROTEIN"] = pc.reindex(sids)

    out["GENE_PANEL"] = panel
    out["N_NONSYN"] = mut.groupby("Tumor_Sample_Barcode").size().reindex(sids).fillna(0).astype(int)
    if panel_sizes_mb:
        out["TMB"] = out["N_NONSYN"] / panel.map(panel_sizes_mb)

    # MUT/HOMDEL/AMP/FUSION columns that are never positive are dropped to keep the table lean
    never = [c for c in out.columns if c.endswith(("_MUT", "_HOMDEL", "_AMP", "_FUSION")) and not out[c].eq(1).any()]
    return out.drop(columns=never).reset_index()


# --- values at a time origin ---------------------------------------------------------------

def baseline_values(values: pd.DataFrame, origins: pd.DataFrame, value_col: str,
                    window: Sequence[int] = (-30, 7), name: Optional[str] = None,
                    time_col: str = "START_DATE", origin_col: str = "ORIGIN_DAY") -> pd.DataFrame:
    """Value of a longitudinal variable (ECOG, a tumour marker, ...) at one or more origins.

    For every (patient, origin) the measurement closest to the origin within ``window``
    (days relative to the origin, bounds included) is taken; on equal distance the one BEFORE
    the origin is preferred (pre-treatment value). Origins may repeat per patient (e.g. one per
    line).

    Returns
    -------
    pd.DataFrame
        PATIENT_ID, ``origin_col``, <name> (default 'BL_<value_col>') and <name>_OFFSET (days
        from origin to the measurement).
    """

    name = name or f"BL_{value_col}"
    o = origins[[ID, origin_col]].drop_duplicates()
    v = values[[ID, time_col, value_col]].dropna(subset=[value_col]).merge(o, on=ID)
    v["_off"] = v[time_col] - v[origin_col]
    v = v[(v["_off"] >= window[0]) & (v["_off"] <= window[1])]
    v["_dist"] = v["_off"].abs() + (v["_off"] > 0) * 0.5
    v = v.sort_values([ID, origin_col, "_dist"], kind="stable").drop_duplicates([ID, origin_col])
    return v[[ID, origin_col, value_col, "_off"]].rename(columns={value_col: name, "_off": f"{name}_OFFSET"}) \
        .reset_index(drop=True)


# --- line level ----------------------------------------------------------------------------

def line_history(lines: pd.DataFrame, diagnosis: Optional[pd.DataFrame] = None,
                 radiation: Optional[pd.DataFrame] = None,
                 agents: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Treatment history before each line.

    Adds PRIOR_LINES, PRIOR_<AGENT> (agent given in any earlier line; ``agents`` = None uses
    every agent that occurs in AGENTS_ALL), PREV_REGIMEN, PREV_TTD_DAYS (if TTD_DAYS exists),
    TFI_DAYS (treatment-free interval since the end of the previous line), DAYS_DX_TO_LINE
    (with ``diagnosis``) and PRIOR_RT (radiation before the line start, with ``radiation``).
    """

    d = lines.sort_values([ID, "LINE_START"], kind="stable").reset_index(drop=True)
    g = d.groupby(ID)
    d["PRIOR_LINES"] = g.cumcount()

    given = d["AGENTS_ALL"].str.split("_")
    agents = sorted({a for l in given for a in l}) if agents is None else list(agents)
    history = []
    for _, idx in g.groups.items():
        seen = set()
        for i in idx:
            history.append((i, frozenset(seen)))
            seen |= set(given[i])
    history = pd.Series(dict(history)).reindex(d.index)
    for a in agents:
        d[f"PRIOR_{column_label(a)}"] = history.map(lambda h, a=a: int(a in h))

    d["PREV_REGIMEN"] = g["REGIMEN"].shift(1)
    if "TTD_DAYS" in d.columns:
        d["PREV_TTD_DAYS"] = g["TTD_DAYS"].shift(1)
    d["TFI_DAYS"] = d["LINE_START"] - g["LINE_END"].shift(1)

    if diagnosis is not None:
        dx_day = diagnosis.set_index(ID)["START_DATE"]
        d["DAYS_DX_TO_LINE"] = d["LINE_START"] - d[ID].map(dx_day)

    if radiation is not None:
        first_rt = radiation.groupby(ID)["START_DATE"].min()
        d["PRIOR_RT"] = (d[ID].map(first_rt) < d["LINE_START"]).astype(int)

    return d


def line_baselines(lines: pd.DataFrame, config: EntityConfig,
                   performance: Optional[pd.DataFrame] = None, markers: Optional[pd.DataFrame] = None,
                   tumor_sites: Optional[pd.DataFrame] = None,
                   ecog_window: Sequence[int] = (-30, 7), marker_window: Sequence[int] = (-30, 7),
                   site_window: Sequence[int] = (-60, 14)) -> pd.DataFrame:
    """Baseline ECOG, tumour markers and metastatic sites at the start of each line.

    Adds BL_ECOG (+ _OFFSET) and ECOG_GE2; per marker of ``config.tumor_markers``
    BL_<M>, LOG10_BL_<M> (values < 0.1 set to 0.1) and, with ``config.marker_uln``,
    BL_<M>_ELEVATED; from the NLP tumour-site timeline SITE_<SITE> (0/1 in ``site_window``),
    N_MET_SITES (without ``config.primary_site_categories``) and SITES_ASSESSED (0 = no
    report in the window; then SITE_* are missing rather than absent).

    Parameters
    ----------
    lines : pd.DataFrame
        One row per line with PATIENT_ID and LINE_START.
    performance, markers : pd.DataFrame
        db['PERFORMANCE'], db['TUMORMARKER'].
    tumor_sites : pd.DataFrame
        Raw ``study.timeline('tumor_sites')`` (one row per site and report).
    """

    d = lines.copy()
    origins = d[[ID, "LINE_START"]].rename(columns={"LINE_START": "ORIGIN_DAY"})

    def add(d, bl):
        return d.merge(bl.rename(columns={"ORIGIN_DAY": "LINE_START"}), on=[ID, "LINE_START"], how="left")

    if performance is not None:
        d = add(d, baseline_values(performance, origins, "ECOG", ecog_window))
        d["ECOG_GE2"] = np.where(d["BL_ECOG"].isna(), np.nan, (d["BL_ECOG"] >= 2).astype(float))

    if markers is not None:
        for test in config.tumor_markers:
            m = column_label(test)
            d = add(d, baseline_values(markers[markers["TEST"] == test], origins, "RESULT", marker_window, f"BL_{m}"))
            d[f"LOG10_BL_{m}"] = np.log10(d[f"BL_{m}"].clip(lower=0.1))
            if test in config.marker_uln:
                d[f"BL_{m}_ELEVATED"] = np.where(d[f"BL_{m}"].isna(), np.nan,
                                                 (d[f"BL_{m}"] > config.marker_uln[test]).astype(float))

    if tumor_sites is not None:
        s = tumor_sites[[ID, "START_DATE", "TUMOR_SITE"]].merge(origins.drop_duplicates(), on=ID)
        off = s["START_DATE"] - s["ORIGIN_DAY"]
        s = s[(off >= site_window[0]) & (off <= site_window[1])]
        tab = pd.crosstab([s[ID], s["ORIGIN_DAY"]], s["TUMOR_SITE"]).clip(upper=1)
        primary = {f"SITE_{column_label(c)}" for c in config.primary_site_categories}
        tab.columns = [f"SITE_{column_label(c)}" for c in tab.columns]
        tab["SITES_ASSESSED"] = 1
        d = add(d, tab.reset_index())
        site_cols = [c for c in tab.columns if c.startswith("SITE_")]
        d["SITES_ASSESSED"] = d["SITES_ASSESSED"].fillna(0).astype(int)
        assessed = d["SITES_ASSESSED"] == 1
        for c in site_cols:
            d[c] = np.where(assessed, d[c].fillna(0), np.nan)
        met = [c for c in site_cols if c not in primary]
        d["N_MET_SITES"] = np.where(assessed, d[met].sum(axis=1), np.nan)

    return d


def marker_kinetics(lines: pd.DataFrame, markers: pd.DataFrame, test: str,
                    baseline_window: Sequence[int] = (-30, 7), min_day: int = 14,
                    landmark_day: int = 56, landmark_window: Sequence[int] = (28, 84),
                    response_threshold: float = -50.0, min_baseline: Optional[float] = None) -> pd.DataFrame:
    """On-treatment course of one tumour marker per line.

    On-treatment values are measured from LINE_START + ``min_day`` until the earlier of
    LINE_END + 14 days and the start of the next line. Adds, with <M> = ``column_label(test)``:

    - <M>_NADIR, <M>_BEST_PCT (largest % decrease from baseline), <M>_N_ONTX
    - <M>_PCT_<landmark_day>D: % change of the value closest to ``landmark_day`` within
      ``landmark_window``
    - <M>_EVALUABLE: baseline >= ``min_baseline`` (e.g. the ULN; None = any baseline) and at
      least one on-treatment value
    - <M>_RESPONSE: BEST_PCT <= ``response_threshold`` (evaluable lines only)

    These variables need time on treatment: relate them to outcomes only in a landmark
    analysis (guarantee-time bias).
    """

    m = column_label(test)
    d = lines.copy()
    mk = markers[markers["TEST"] == test][[ID, "START_DATE", "RESULT"]].dropna()
    origins = d[[ID, "LINE_START"]].rename(columns={"LINE_START": "ORIGIN_DAY"})
    bl = baseline_values(mk, origins, "RESULT", baseline_window, "_BL").rename(columns={"ORIGIN_DAY": "LINE_START"})

    if "NEXT_LINE_START" not in d.columns:
        d["NEXT_LINE_START"] = d.sort_values([ID, "LINE_START"]).groupby(ID)["LINE_START"].shift(-1)

    win = d[[ID, "LINE_START", "LINE_END", "NEXT_LINE_START"]].merge(bl[[ID, "LINE_START", "_BL"]],
                                                                   on=[ID, "LINE_START"], how="left")
    win["_upper"] = np.fmin(win["LINE_END"] + 14, win["NEXT_LINE_START"].fillna(np.inf))
    x = win.merge(mk, on=ID)
    x["_t"] = x["START_DATE"] - x["LINE_START"]
    x = x[(x["_t"] >= min_day) & (x["START_DATE"] <= x["_upper"])]
    x["_pct"] = 100 * (x["RESULT"] - x["_BL"]) / x["_BL"].where(x["_BL"] > 0)

    agg = x.groupby([ID, "LINE_START"]).agg(NADIR=("RESULT", "min"), BEST_PCT=("_pct", "min"),
                                            N_ONTX=("RESULT", "size")).reset_index()
    lm = x[(x["_t"] >= landmark_window[0]) & (x["_t"] <= landmark_window[1])].copy()
    lm["_d"] = (lm["_t"] - landmark_day).abs()
    lm = lm.sort_values("_d", kind="stable").drop_duplicates([ID, "LINE_START"])[[ID, "LINE_START", "_pct"]]
    lm = lm.rename(columns={"_pct": f"PCT_{landmark_day}D"})

    res = win[[ID, "LINE_START", "_BL"]].merge(agg, on=[ID, "LINE_START"], how="left") \
                                        .merge(lm, on=[ID, "LINE_START"], how="left")
    res["N_ONTX"] = res["N_ONTX"].fillna(0).astype(int)
    ok_bl = res["_BL"].notna() & (res["_BL"] >= min_baseline if min_baseline is not None else True)
    res["EVALUABLE"] = (ok_bl & (res["N_ONTX"] > 0)).astype(int)
    res["RESPONSE"] = np.where(res["EVALUABLE"] == 1, (res["BEST_PCT"] <= response_threshold).astype(float), np.nan)

    keep = ["NADIR", "BEST_PCT", "N_ONTX", f"PCT_{landmark_day}D", "EVALUABLE", "RESPONSE"]
    res = res.drop(columns="_BL").rename(columns={c: f"{m}_{c}" for c in keep})
    return d.merge(res, on=[ID, "LINE_START"], how="left")


def build_line_table(line_endpoints: pd.DataFrame, cohort, db: Dict[str, pd.DataFrame], study,
                     genomics: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """One analysis-ready row per line: endpoints + history, patient, genomic and baseline covariates.

    Parameters
    ----------
    line_endpoints : pd.DataFrame
        ``endpoints.line_endpoints`` output.
    cohort : Cohort
    db : dict
        ``tables.build_database`` output (PERFORMANCE, TUMORMARKER, RADIATION are used).
    study : ChordStudy
        For the NLP tumour-site timeline.
    genomics : pd.DataFrame, optional
        ``genomic_features`` output; merged via the first sequenced sample (SAMPLE_ID).

    Also adds SAMPLE_BEFORE_LINE: 1 if the sequenced tissue was obtained on or before the line
    start (SAMPLE_ACQ_DAY), 0 if later (genomics then describe the tumour after that line),
    missing if the acquisition day is unknown.
    """

    config = cohort.config
    d = line_history(line_endpoints, cohort.diagnosis, db.get("RADIATION"), config.treatment.agents)
    d = d.merge(patient_covariates(cohort, study), on=ID, how="left")
    if genomics is not None:
        d = d.merge(genomics, on="SAMPLE_ID", how="left")

    sites = cohort.restrict(study.timeline("tumor_sites"), log=False)
    d = line_baselines(d, config, db.get("PERFORMANCE"), db.get("TUMORMARKER"), sites)
    if "TUMORMARKER" in db:
        for test in config.tumor_markers:
            d = marker_kinetics(d, db["TUMORMARKER"], test, min_baseline=config.marker_uln.get(test))

    d["SAMPLE_BEFORE_LINE"] = np.where(d["SAMPLE_ACQ_DAY"].isna(), np.nan,
                                       (d["SAMPLE_ACQ_DAY"] <= d["LINE_START"]).astype(float))
    return d
