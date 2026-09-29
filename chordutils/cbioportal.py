"""Driver annotations from cBioPortal "alterations across samples" exports.

The raw MSK-CHORD genomic files (used by ``covariates.genomic_features``) contain no
functional annotation: every nonsynonymous mutation counts, including variants of unknown
significance. cBioPortal annotates alterations as "driver" (by default based on OncoKB and
cancer hotspots) and lets you download, for a gene query, one table per study via
*Download -> Alterations across samples*. This module turns such an export into driver
matrices.

Export format
-------------
One row per sample with ``Study ID``, ``Sample ID``, ``Patient ID``, ``Altered`` and, per
queried gene, a summary column ``<GENE>`` plus typed columns ``<GENE>: MUT``,
``<GENE>: AMP``, ``<GENE>: HOMDEL`` and ``<GENE>: FUSION`` (some studies lack a type, e.g. no
FUSION column without structural-variant data). A cell holds ``no alteration``,
``not profiled`` or the events joined by ', ', each optionally followed by ``(driver)``,
e.g. ``R175H (driver), P322Hfs*23``.

Checked on twelve exports (MSK-CHORD, MSK-MET, GENIE BPC CRC, TCGA CRC/STAD/EAC/BRCA/GBM/LUAD/
LUSC, CPTAC COAD), 2026-09-30:

- The summary column is always the union of the typed columns, so only the typed columns are
  used here and the type of every event is known (no guessing from the text).
- ``not profiled`` differs between types: e.g. samples without mutation data but with copy
  number data show ``no alteration`` in the summary column. Profiling is therefore tracked
  per gene AND type; 'not profiled' is missing, never 'wild type'.
- Structural variants are written in many forms ('X-Y fusion', 'X-Y', 'APC-intragenic',
  'Deletion within transcript: mid-exon', ...) and are never labelled '(driver)'. They are
  reported separately (``<GENE>_SV``), not as drivers.

The exports are derived data of the respective studies; keep them out of version control
when the study licence forbids redistribution (MSK-CHORD: CC BY-NC-ND 4.0).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

TYPES = ("MUT", "AMP", "HOMDEL", "FUSION")
DRIVER_TAG = " (driver)"
NO_ALTERATION = "no alteration"
NOT_PROFILED = "not profiled"
_META = {"Study ID": "STUDY_ID", "Sample ID": "SAMPLE_ID", "Patient ID": "PATIENT_ID"}


@dataclass
class AlterationExport:
    """Parsed cBioPortal export.

    Attributes
    ----------
    samples : pd.DataFrame
        SAMPLE_ID, PATIENT_ID, STUDY_ID in file order.
    events : pd.DataFrame
        One row per alteration: SAMPLE_ID, GENE, TYPE (MUT/AMP/HOMDEL/FUSION), EVENT (e.g.
        'G12D', 'AMP', 'TRAP1-CREBBP fusion'), DRIVER (bool).
    profiled : pd.DataFrame
        Boolean, index SAMPLE_ID, columns MultiIndex (GENE, TYPE): was the gene profiled for
        this alteration type in this sample. Types missing from the export are False.
    """

    samples: pd.DataFrame
    events: pd.DataFrame
    profiled: pd.DataFrame

    @property
    def genes(self) -> list:
        return list(dict.fromkeys(self.profiled.columns.get_level_values(0)))

    def __repr__(self):
        n_drv = int(self.events["DRIVER"].sum())
        return (f"AlterationExport({len(self.samples)} samples, {len(self.genes)} genes, "
                f"{len(self.events)} events, {n_drv} drivers)")


def _split_events(cell: str):
    """'R175H (driver), V157F' -> [('R175H', True), ('V157F', False)]; none for the status values."""

    if cell in (NO_ALTERATION, NOT_PROFILED) or pd.isna(cell):
        return []
    out = []
    for ev in cell.split(", "):
        driver = ev.endswith(DRIVER_TAG)
        out.append((ev[: -len(DRIVER_TAG)] if driver else ev, driver))
    return out


def read_alteration_export(path: Union[str, Path]) -> AlterationExport:
    """Read a cBioPortal 'alterations across samples' export (see module docstring).

    Parameters
    ----------
    path : str or Path
        The downloaded .tsv file.

    Returns
    -------
    AlterationExport
    """

    raw = pd.read_table(path, dtype=str, keep_default_na=False)
    missing = [c for c in _META if c not in raw.columns]
    if missing:
        raise ValueError(f"Not a cBioPortal alteration export, missing columns: {missing}")

    samples = raw[list(_META)].rename(columns=_META)[["SAMPLE_ID", "PATIENT_ID", "STUDY_ID"]]
    if samples["SAMPLE_ID"].duplicated().any():
        raise ValueError("Sample IDs are not unique (export from several studies?)")
    sids = samples["SAMPLE_ID"].to_numpy()

    typed = [c for c in raw.columns if ": " in c and c.split(": ", 1)[1] in TYPES]
    genes = list(dict.fromkeys(c.split(": ", 1)[0] for c in typed))
    if not genes:
        raise ValueError("No typed columns ('<GENE>: MUT', ...) found; re-download the export from cBioPortal")

    rows, profiled = [], {}
    for g in genes:
        for t in TYPES:
            col = f"{g}: {t}"
            if col not in raw.columns:
                profiled[(g, t)] = np.zeros(len(raw), dtype=bool)
                continue
            values = raw[col].to_numpy()
            profiled[(g, t)] = values != NOT_PROFILED
            for sid, cell in zip(sids, values):
                for ev, drv in _split_events(cell):
                    rows.append((sid, g, t, ev, drv))

    events = pd.DataFrame(rows, columns=["SAMPLE_ID", "GENE", "TYPE", "EVENT", "DRIVER"])
    prof = pd.DataFrame(profiled, index=pd.Index(sids, name="SAMPLE_ID"))
    prof.columns = pd.MultiIndex.from_tuples(prof.columns, names=["GENE", "TYPE"])
    return AlterationExport(samples.reset_index(drop=True), events, prof)


def panel_genes_from_export(export: AlterationExport, gene_panel: pd.Series,
                            alteration_type: str = "MUT") -> Dict[str, list]:
    """Genes on each sequencing panel, as observed in an export.

    A gene counts as being on a panel if it was profiled (``alteration_type``) in EVERY sample
    of that panel in the export. The result can be passed as ``panel_genes`` to
    ``covariates.genomic_features`` (the MSK-CHORD download itself has no panel gene lists).
    Only genes of the export's gene query are covered.

    Parameters
    ----------
    export : AlterationExport
    gene_panel : pd.Series
        Panel per sample, index SAMPLE_ID (e.g. ``data_gene_panel_matrix.txt``, column
        'mutations').
    alteration_type : str
        Profiling of which type decides.

    Returns
    -------
    dict of {panel: list of genes}
        Panels without any sample in the export are omitted.
    """

    prof = export.profiled.xs(alteration_type, axis=1, level="TYPE")
    panel = gene_panel.reindex(prof.index)
    out = {}
    for p, sub in prof.groupby(panel):
        out[p] = [g for g in prof.columns if sub[g].all()]
    return out


def driver_event_matrix(export: AlterationExport, types: Sequence[str] = ("MUT", "AMP", "HOMDEL"),
                        min_samples: int = 1, order: bool = True) -> pd.DataFrame:
    """Samples x driver events ('GENE:EVENT', e.g. 'KRAS:G12D', 'ERBB2:AMP').

    Parameters
    ----------
    export : AlterationExport
    types : sequence of {'MUT', 'AMP', 'HOMDEL', 'FUSION'}
        Alteration types included. FUSION events are never labelled as drivers in the exports
        checked so far; see ``driver_gene_matrix`` for structural variants.
    min_samples : int
        Events present in fewer samples are dropped.
    order : bool
        Order genes by their total number of driver events and, within a gene, events by
        frequency (both descending), e.g. for oncoprints.

    Returns
    -------
    pd.DataFrame
        Index SAMPLE_ID; 1 = event present, 0 = absent, missing = the gene was not profiled
        for the event's type in this sample.
    """

    ev = export.events[export.events["DRIVER"] & export.events["TYPE"].isin(types)]
    ev = ev.assign(COLUMN=ev["GENE"] + ":" + ev["EVENT"]).drop_duplicates(["SAMPLE_ID", "COLUMN"])

    counts = ev.groupby(["GENE", "COLUMN"]).size().rename("n").reset_index()
    counts = counts[counts["n"] >= min_samples]
    if order:
        gene_total = counts.groupby("GENE")["n"].sum()
        counts = counts.assign(_g=counts["GENE"].map(gene_total)) \
                       .sort_values(["_g", "GENE", "n", "COLUMN"], ascending=[False, True, False, True])

    columns = pd.Index(counts["COLUMN"].tolist())
    index = export.profiled.index
    values = np.zeros((len(index), len(columns)))
    hits = ev[ev["COLUMN"].isin(columns)]
    values[index.get_indexer(hits["SAMPLE_ID"]), columns.get_indexer(hits["COLUMN"])] = 1.0
    m = pd.DataFrame(values, index=index, columns=columns)

    event_type = ev.drop_duplicates("COLUMN").set_index("COLUMN")[["GENE", "TYPE"]]
    for c in columns:
        g, t = event_type.loc[c]
        m[c] = m[c].where(export.profiled[(g, t)])
    return m


def driver_gene_matrix(export: AlterationExport, types: Sequence[str] = ("MUT", "AMP", "HOMDEL"),
                       gene_groups: Optional[Dict[str, Sequence[str]]] = None,
                       fusions: bool = True) -> pd.DataFrame:
    """One row per sample: driver status per gene (and gene group), plus structural variants.

    Columns
    -------
    <GENE>_DRIVER
        1 if the gene has a driver event of one of ``types``; 0 if it was profiled for ALL of
        ``types`` and has none; missing otherwise (e.g. no mutation data for the sample).
    <GROUP>_DRIVER, <GROUP>_N_DRIVER
        For ``gene_groups`` (e.g. ``config.gene_groups``): 1 if any gene of the group has a
        driver, 0 if all genes are 0, else missing; N_DRIVER = number of genes with a driver.
    <GENE>_SV (with ``fusions=True``)
        1 if the export lists any structural variant for the gene (fusion, intragenic
        deletion/duplication, ...; the exports carry no driver label for these), 0 if
        profiled without one, missing if not profiled. Inspect the individual events in
        ``export.events.query("TYPE == 'FUSION'")`` before using a column.

    Returns
    -------
    pd.DataFrame
        SAMPLE_ID + the columns above. Column names do not overlap with
        ``covariates.genomic_features`` (``_ALT``, ``_N``, ``_FUSION``), so both can be merged on SAMPLE_ID
        and passed to ``covariates.build_line_table``.
    """

    prof = export.profiled
    ev = export.events
    drv = ev[ev["DRIVER"] & ev["TYPE"].isin(types)]
    out = {}

    for g in export.genes:
        profiled_all = prof[[(g, t) for t in types]].all(axis=1)
        has = prof.index.isin(drv.loc[drv["GENE"] == g, "SAMPLE_ID"])
        out[f"{g}_DRIVER"] = np.where(has, 1.0, np.where(profiled_all, 0.0, np.nan))

    res = pd.DataFrame(out, index=prof.index)

    for grp, gl in (gene_groups or {}).items():
        cols = [f"{g}_DRIVER" for g in gl if f"{g}_DRIVER" in res.columns]
        absent = [g for g in gl if f"{g}_DRIVER" not in res.columns]
        if absent:
            raise ValueError(f"Genes of group {grp!r} not in the export: {absent}")
        sub = res[cols]
        res[f"{grp}_N_DRIVER"] = sub.sum(axis=1)
        res[f"{grp}_DRIVER"] = np.where(sub.eq(1).any(axis=1), 1.0, np.where(sub.notna().all(axis=1), 0.0, np.nan))

    if fusions:
        sv = ev[ev["TYPE"] == "FUSION"]
        for g in export.genes:
            has = prof.index.isin(sv.loc[sv["GENE"] == g, "SAMPLE_ID"])
            res[f"{g}_SV"] = np.where(has, 1.0, np.where(prof[(g, "FUSION")], 0.0, np.nan))

    return res.reset_index()
