# Changelog

## Unreleased

### Changed
- `cbioportal` moved to the new package `cbiokit` (`cbiokit.alterations`); `chordutils.cbioportal` is
  now a thin alias, so existing code keeps working. `cbiokit` is a new dependency; its tests replace
  `tests/test_cbioportal.py`.

## 0.3.0 (2026-09-30)

### Added
- `markers.marker_course`: values of a tumour marker (or any longitudinal variable, e.g. ECOG)
  within each line of therapy, one row per value and line. Window from `pre_days` before the
  line start to `post_days` after its end (default 21/21), cut at the start of the next line;
  baseline = last value up to the line start (`baseline='all'` keeps all pre-treatment
  values); optional baseline threshold (`baseline_above`, e.g. the ULN); same-day values
  combined by mean, median or first; minimum number of values; every exclusion logged.
- `markers.to_tumgr`: input for `tumgr::gdrate` (name/date/size). tumgr requires numeric
  identifiers, so lines get running numbers; the mapping is returned.
  Checked with tumgr 0.0.4: zeros during follow-up and courses with two values are accepted.

## 0.2.0 (2026-09-30)

### Added
- `cbioportal`: driver (OncoKB) annotations from cBioPortal "alterations across samples"
  exports. `read_alteration_export` (one row per event with gene, type MUT/AMP/HOMDEL/FUSION
  and driver flag, plus profiling per gene and type), `driver_event_matrix` (samples x
  'GENE:EVENT'), `driver_gene_matrix` (`<GENE>_DRIVER`, `<GROUP>_DRIVER`/`_N_DRIVER`,
  `<GENE>_SV`), `panel_genes_from_export` (panel gene content for `genomic_features`).
  Developed from an earlier notebook routine (TCGA CRC); on that export the driver values are
  identical wherever the gene was profiled. Differences by design: "not profiled" is missing
  instead of 0 (the old routine counted 62 TCGA samples without mutation data as wild type),
  samples instead of patients as index, profiling per alteration type, structural variants
  (never labelled as drivers) in their own column `<GENE>_SV`. Checked on twelve exports (MSK-CHORD,
  MSK-MET, GENIE BPC CRC, TCGA, CPTAC).

### Changed
- `covariates.genomic_features`: `panel_genes` masks only genes listed for at least one panel;
  genes of unknown coverage (e.g. outside an export's gene query) stay unmasked.

## 0.1.0 (2026-09-29)

First release as a package, developed from the single module `chordutils.py` of the
MSK-CHORD PDAC analysis and two follow-up analysis modules (chordendpoints, chordcovariates).
The full PDAC analysis database is reproduced exactly by the package (verified against the
original notebook output), apart from the corrections below.

### Added
- `io`, `config`, `entities`, `cohort`, `timeline`, `attrition`: study access, entity
  configuration, cohort selection with a logged filter flow, alignment to the diagnosis.
- `tables`: analysis tables per timeline and `build_database`.
- `lines`, `endpoints`, `survival`, `covariates`: entity-agnostic parts of the follow-up
  modules (lines of therapy, TTD/TTNT/rwPFS/OS, delayed entry, time-varying exposures,
  landmark, Cox screen via lifelines, covariates). PDAC-specific heuristics (resection,
  recurrence, DFS, treatment setting, KRAS allele classes) were not included.

### Fixed
- `pivot_therapies`: regimen names could keep bookkeeping digits (e.g. `GEMCITABINE0-`) with
  >= 10 agent rows per patient; agent names ending in digits (e.g. `SODIUM IODIDE I-131`)
  would have been truncated. Unsorted input could produce overlapping episodes; empty input
  raised an error.
- `validate_event` rejected NumPy integers; `np.in1d` replaced by `np.isin`; error messages
  and debug prints in the `windows` helpers.
- PDAC config: tumour markers are kept from 90 days before diagnosis (the original code kept
  them only from day 1 after diagnosis).
- Sample timing relative to a line uses the tissue acquisition day, not the sequencing day.
- Survival CIs use z = 1.95996 (as lifelines and R).

### Design decisions
- Entity diagnoses are matched by ICD-O-3 topography code rather than site text.
- Several matching registry diagnoses per patient: earliest as time origin (logged) or stop.
- Genomic calls: missing copy-number data stay missing; off-panel genes can be masked with
  `panel_genes` (panel gene lists are not in the MSK-CHORD download).
- No default panel sizes for TMB.
