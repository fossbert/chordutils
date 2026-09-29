# chordutils

Analysis helpers for [MSK-CHORD](https://www.cbioportal.org/study/summary?id=msk_chord_2024)
(Jee et al., *Nature* 2024), the MSK clinicogenomic real-world data set on cBioPortal
(NSCLC, breast, colorectal, prostate and pancreatic cancer).

The package turns the cBioPortal study files into analysis tables for one tumour entity:
cohort selection, alignment of all timelines to the diagnosis, regimen episodes and lines of
therapy, real-world endpoints (TTD, TTNT, rwPFS, OS) and covariates. The functions are the
same for every entity; everything entity-specific is written down in one `EntityConfig`.
Every filter step is logged with its reason and the number of rows and patients before and
after, so a cohort can be traced step by step (e.g. for a CONSORT diagram).

**Data are not included.** MSK-CHORD is distributed under CC BY-NC-ND 4.0; download the study
folder `msk_chord_2024` from cBioPortal. The MIT license of this repository covers the code
only.

## Install

```bash
pip install -e .              # numpy, pandas
pip install -e '.[survival]'  # + lifelines (Cox screen in survival.association_screen)
pip install -e '.[test]'
```

## Quick start

```python
import chordutils as cu

study = cu.ChordStudy("msk_chord_2024")               # cBioPortal study folder
cohort = cu.select_cohort(study, cu.entities.PDAC)    # samples, patients, diagnosis
cohort.log                                            # every filter step with counts

db = cu.build_database(cohort, study)                 # TREATMENT, SURGERY, STAGING, ...
lines = cu.lines.build_lines_of_therapy(db["TREATMENT"], cohort.config.lines)
E = cu.endpoints.line_endpoints(lines,
                                cu.endpoints.collapse_assessments(db["STAGING"]),
                                cu.endpoints.patient_followup(cohort.patients))
T = cu.covariates.build_line_table(E, cohort, db, study)   # one analysis row per line
```

## Layout

| Module | Contents |
| --- | --- |
| `io` | `ChordStudy` (cached access to all study files, `timeline(name)`, `dictionary()` = column descriptions from the cBioPortal headers), `read_cbio_table`, `pickle_transfer` |
| `config` | `EntityConfig`, `TreatmentRules`, `LineRules`, `icdo_topography_pattern` |
| `entities` | ready-made configs: `PDAC` |
| `cohort` | `select_cohort` -> `Cohort` (samples, patients, diagnosis, log; `restrict`, `align`) |
| `timeline` | `align_to_diagnosis` (time from diagnosis + window), `drop_constant_columns` |
| `attrition` | `AttritionLog` |
| `therapy` | `pivot_therapies`: one row per agent -> non-overlapping regimen episodes |
| `tables` | `treatment_episodes`, `surgeries`, `radiation`, `imaging_assessments`, `tumor_markers`, `performance_status`, `build_database` |
| `lines` | `build_lines_of_therapy`, `classify_regimen` |
| `endpoints` | `patient_followup`, `collapse_assessments`, `line_endpoints` (TTD, TTNT, rwPFS, OS), `os_base` |
| `survival` | `km_estimate`/`km_median` (delayed entry), `summarize_endpoint`, `exposure_days`, `to_counting_process`, `landmark`, `association_screen`, `benjamini_hochberg` |
| `covariates` | `patient_covariates`, `genomic_features`, `baseline_values`, `line_history`, `line_baselines`, `marker_kinetics`, `build_line_table` |
| `windows` | legacy look-ups around an event: `find_stagings`, `find_ps`, `find_markers` |

## Conventions and caveats

- **Time axis.** All timeline days are relative to the patient's first sequenced sample
  (day 0). `OS_MONTHS` uses months of 365/12 days from the same day 0.
- **Left truncation.** Patients are only in MSK-CHORD because a sample was sequenced. Endpoint
  tables carry `ENTRY` (days from the time origin to day 0) for delayed entry in OS analyses;
  for TTD/rwPFS/TTNT use an inception cohort (small `ENTRY`).
- **Imaging.** The NLP timelines have no report identifier; use
  `endpoints.collapse_assessments` (one assessment per day) for endpoints.
- **Genomics.** The gene content of the MSK-IMPACT panel versions is not part of the download,
  and `data_cna.txt` reports 0 for genes outside a sample's panel. Pass `panel_genes` to
  `genomic_features` to mask off-panel genes, or account for `GENE_PANEL`.
- **Sample timing.** `SEQ_DATE` is the day of sequencing; the day the tissue was obtained is
  `SAMPLE_ACQ_DAY` (specimen-surgery timeline).

## Adding an entity

Create `chordutils/entities/<name>.py` with an `EntityConfig` and import it in
`entities/__init__.py`. Decide and document in the config:

1. `sample_filters` (e.g. `{"CANCER_TYPE": ["Breast Cancer"]}`) and `diagnosis_pattern`
   (`icdo_topography_pattern(["C50"])`), `multiple_diagnoses` ('first' or 'error' for review);
2. `windows` per timeline relative to the diagnosis;
3. `treatment` (subtypes incl. Hormone for breast/prostate, cleaning rules) and `lines`
   (protocol names, equivalent agents);
4. `tumor_markers`, `marker_uln`, `genes`, `gene_groups`, `primary_site_categories`.

Check `cohort.log` after the first run: unexpected losses show up there.

## Tests

```bash
pytest
```

The tests use small synthetic data. Regression tests against the full MSK-CHORD data live in
the analysis project, since the data cannot be distributed.
