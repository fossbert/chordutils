"""Tests for chordutils.cbioportal on a synthetic 'alterations across samples' export.

The export mimics the cases found in real exports: several events per cell, driver and
non-driver events, 'not profiled' that differs between types, structural variants in
various spellings, a study without FUSION columns for one gene, two samples of one patient.
"""

import numpy as np
import pandas as pd
import pytest

from chordutils.cbioportal import (driver_event_matrix, driver_gene_matrix, panel_genes_from_export,
                                   read_alteration_export)

NA, NP = "no alteration", "not profiled"

ROWS = [
    # sample, patient, KRAS (MUT, AMP, HOMDEL, FUSION), TP53 (MUT, AMP, HOMDEL, FUSION), ERBB2 (MUT, AMP, HOMDEL)
    ("S1", "P1", "G12D (driver)", NA, NA, NA,
                 "R175H (driver), P322Hfs*23", NA, NA, "TP53-intragenic",
                 NA, "AMP (driver)", NA),
    ("S2", "P1", "G12D (driver)", "AMP", NA, NA,          # second sample of P1; non-driver AMP
                 NA, NA, "HOMDEL (driver)", NA,
                 "S310F", NA, NA),
    ("S3", "P2", NP, NA, NA, NP,                            # no mutation data, CNA profiled
                 NP, NA, NA, NP,
                 NP, NA, NA),
    ("S4", "P3", "G12V (driver)", NA, NA, "KRAS-CDH1 fusion, Deletion within transcript: mid-exon",
                 "MUTATED (driver)", NA, NA, NA,
                 NA, NA, NA),
    ("S5", "P4", NP, NP, NP, NP,                            # nothing profiled
                 NP, NP, NP, NP,
                 NP, NP, NP),
]


def _summary(*cells):
    ev = [c for c in cells if c not in (NA, NP)]
    if ev:
        return ", ".join(ev)
    return NP if all(c == NP for c in cells) else NA


@pytest.fixture
def export(tmp_path):
    records = []
    for s, p, *v in ROWS:
        kras, tp53, erbb2 = v[0:4], v[4:8], v[8:11]
        altered = int(any(c not in (NA, NP) for c in v))
        rec = {"Study ID": "demo", "Sample ID": s, "Patient ID": p, "Altered": altered,
               "KRAS": _summary(*kras), "TP53": _summary(*tp53), "ERBB2": _summary(*erbb2)}
        for g, vals, types in [("KRAS", kras, "MUT AMP HOMDEL FUSION"), ("TP53", tp53, "MUT AMP HOMDEL FUSION"),
                               ("ERBB2", erbb2, "MUT AMP HOMDEL")]:
            rec.update({f"{g}: {t}": x for t, x in zip(types.split(), vals)})
        records.append(rec)
    f = tmp_path / "alterations_across_samples.tsv"
    pd.DataFrame(records).to_csv(f, sep="\t", index=False)
    return read_alteration_export(f)


def test_read(export):
    assert export.genes == ["KRAS", "TP53", "ERBB2"]
    assert export.samples["PATIENT_ID"].tolist() == ["P1", "P1", "P2", "P3", "P4"]
    ev = export.events.set_index(["SAMPLE_ID", "GENE", "TYPE", "EVENT"])["DRIVER"]
    assert ev[("S1", "TP53", "MUT", "R175H")] and not ev[("S1", "TP53", "MUT", "P322Hfs*23")]
    assert not ev[("S2", "KRAS", "AMP", "AMP")]
    assert ev[("S4", "TP53", "MUT", "MUTATED")]
    assert ("S4", "KRAS", "FUSION", "Deletion within transcript: mid-exon") in ev.index
    # missing FUSION column for ERBB2 -> never profiled for fusions
    assert not export.profiled[("ERBB2", "FUSION")].any()
    assert export.profiled[("KRAS", "AMP")].tolist() == [True, True, True, True, False]


def test_read_rejects_other_tables(tmp_path):
    f = tmp_path / "x.tsv"
    pd.DataFrame({"a": [1]}).to_csv(f, sep="\t", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        read_alteration_export(f)


def test_driver_event_matrix(export):
    m = driver_event_matrix(export)
    # genes by total driver events (KRAS 3, TP53 3 -> tie broken alphabetically, ERBB2 1),
    # events within a gene by frequency, then alphabetically
    assert list(m.columns) == ["KRAS:G12D", "KRAS:G12V", "TP53:HOMDEL", "TP53:MUTATED", "TP53:R175H", "ERBB2:AMP"]
    assert m.loc["S1"].tolist() == [1, 0, 0, 0, 1, 1]
    assert m.loc["S2"].tolist() == [1, 0, 1, 0, 0, 0]          # non-driver AMP and S310F not counted
    s3 = m.loc["S3"]
    assert s3[["KRAS:G12D", "TP53:R175H"]].isna().all()        # no mutation data -> missing, not 0
    assert s3[["TP53:HOMDEL", "ERBB2:AMP"]].tolist() == [0, 0]  # CNA profiled -> 0
    assert m.loc["S5"].isna().all()


def test_driver_event_matrix_min_samples(export):
    assert list(driver_event_matrix(export, min_samples=2).columns) == ["KRAS:G12D"]


def test_driver_gene_matrix(export):
    g = driver_gene_matrix(export, gene_groups={"CORE": ["KRAS", "TP53"]}).set_index("SAMPLE_ID")
    assert g.loc[["S1", "S2", "S4"], "KRAS_DRIVER"].tolist() == [1, 1, 1]
    assert g.loc[["S3", "S5"], "KRAS_DRIVER"].isna().all()      # S3: MUT not profiled; S5: nothing
    assert g["ERBB2_DRIVER"].tolist()[:2] == [1, 0]            # S2: S310F is not a driver
    assert g.loc["S1", "CORE_N_DRIVER"] == 2 and g.loc["S1", "CORE_DRIVER"] == 1
    assert np.isnan(g.loc["S5", "CORE_DRIVER"])
    # structural variants: own column, no driver label needed
    assert g.loc[["S1", "S2", "S4"], "KRAS_SV"].tolist() == [0, 0, 1]
    assert np.isnan(g.loc["S3", "KRAS_SV"])                # fusions not profiled in S3
    assert g.loc["S1", "TP53_SV"] == 1
    assert g["ERBB2_SV"].isna().all()                      # no FUSION column in the export


def test_driver_gene_matrix_types_and_groups(export):
    g = driver_gene_matrix(export, types=("AMP", "HOMDEL"), fusions=False).set_index("SAMPLE_ID")
    assert g.loc["S3"].tolist() == [0, 0, 0]                   # CNA-only definition: S3 is evaluable
    assert not any(c.endswith("_SV") for c in g.columns)
    with pytest.raises(ValueError, match="not in the export"):
        driver_gene_matrix(export, gene_groups={"HRR": ["BRCA2"]})


def test_panel_genes_from_export(export):
    panel = pd.Series({"S1": "NEW", "S2": "NEW", "S3": "OLD", "S4": "NEW", "S5": "OLD"})
    assert panel_genes_from_export(export, panel) == {"NEW": ["KRAS", "TP53", "ERBB2"], "OLD": []}
