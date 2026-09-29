"""Selection of an entity cohort from MSK-CHORD.

``select_cohort`` applies an ``EntityConfig`` to the study files and returns a ``Cohort`` with
the sample, patient and diagnosis tables of the entity plus the log of every filter step.

Steps (each recorded in ``Cohort.log``)
---------------------------------------
1. Samples: apply ``config.sample_filters`` one after another.
2. Sequencing day: add SEQ_DATE (day of the sample in the specimen timeline) to each sample.
3. Patients: patient table restricted to patients with at least one selected sample;
   OS_STATUS converted from '1:DECEASED' / '0:LIVING' to 1 / 0.
4. Diagnosis: registry diagnoses matching ``config.diagnosis_pattern``; with several matching
   diagnoses per patient ``config.multiple_diagnoses`` decides.
5. Optionally, patients without a matching diagnosis are removed from all tables.
"""

from dataclasses import dataclass, field

import pandas as pd

from .attrition import AttritionLog
from .config import EntityConfig
from .io import ChordStudy
from .timeline import align_to_diagnosis, drop_constant_columns

ID = "PATIENT_ID"


@dataclass
class Cohort:
    """Tables of one entity cohort.

    Attributes
    ----------
    config : EntityConfig
        The configuration the cohort was built with.
    samples : pd.DataFrame
        Selected samples (``data_clinical_sample.txt`` + SEQ_DATE).
    patients : pd.DataFrame
        Patient table of the cohort (``data_clinical_patient.txt``), OS_STATUS as 0/1.
    diagnosis : pd.DataFrame
        One row per patient: the entity's registry diagnosis (START_DATE = day of diagnosis).
    log : AttritionLog
        All filter steps with reasons and counts.
    """

    config: EntityConfig
    samples: pd.DataFrame
    patients: pd.DataFrame
    diagnosis: pd.DataFrame
    log: AttritionLog = field(default_factory=AttritionLog)

    @property
    def patient_ids(self) -> pd.Index:
        """Identifiers of all patients in the cohort."""

        return pd.Index(self.patients[ID].unique(), name=ID)

    def restrict(self, d: pd.DataFrame, table: str = "", log: bool = True) -> pd.DataFrame:
        """Keep only rows of cohort patients (e.g. of a timeline table); logged by default."""

        mask = d[ID].isin(self.patient_ids)
        if log:
            return self.log.filter(d, mask, table, "cohort patients", f"{self.config.name} cohort")
        return d[mask]

    def align(self, events: pd.DataFrame, timeline: str, restrict: bool = True, **kwargs) -> pd.DataFrame:
        """Restrict events to the cohort and to the configured window around the diagnosis.

        Shorthand for ``restrict`` (skipped with ``restrict=False`` for tables that are
        already restricted) followed by ``align_to_diagnosis`` with ``config.window(timeline)``;
        keyword arguments are passed on to ``align_to_diagnosis``.
        """

        if restrict:
            events = self.restrict(events, table=timeline)
        return align_to_diagnosis(events, self.diagnosis, window=self.config.window(timeline),
                                  log=self.log, table=timeline, **kwargs)

    def __repr__(self):
        return (f"Cohort({self.config.name}: {len(self.patient_ids)} patients, "
                f"{len(self.samples)} samples, {len(self.log.steps)} logged steps)")


def parse_os_status(status: pd.Series) -> pd.Series:
    """Convert cBioPortal OS_STATUS ('1:DECEASED', '0:LIVING') to integers 1 / 0."""

    if pd.api.types.is_numeric_dtype(status):
        return status
    code = status.str.split(":").str[0]
    return code.astype(int) if code.notna().all() else pd.to_numeric(code).astype("Int64")


def select_diagnoses(diagnosis: pd.DataFrame, config: EntityConfig, log: AttritionLog,
                     table: str = "diagnosis") -> pd.DataFrame:
    """Registry diagnoses of the entity, one row per patient (see ``EntityConfig``)."""

    match = diagnosis["DX_DESCRIPTION"].str.contains(config.diagnosis_pattern, regex=True, na=False)
    dx = log.filter(diagnosis, match, table, "entity diagnosis",
                    f"DX_DESCRIPTION matches {config.diagnosis_pattern!r}")

    n_multi = int(dx[ID].duplicated().sum())
    multi_ids = dx.loc[dx[ID].duplicated(keep=False), ID].unique()

    if n_multi and config.multiple_diagnoses == "error":
        raise ValueError(f"{len(multi_ids)} patients have more than one matching diagnosis, e.g. "
                         f"{list(multi_ids[:5])}. Review them or set multiple_diagnoses='first'.")

    first_idx = dx.sort_values([ID, "START_DATE"], kind="stable").drop_duplicates(ID).index
    first = dx[dx.index.isin(first_idx)]  # keeps the original row order
    log.record(dx, first, table, "one diagnosis per patient",
               "earliest matching diagnosis is the time origin",
               detail=f"{len(multi_ids)} patients with several matching diagnoses" if n_multi else "")
    return first


def select_cohort(study: ChordStudy,
                  config: EntityConfig,
                  require_diagnosis: bool = True,
                  drop_constant: bool = False) -> Cohort:
    """Build the cohort of one entity (see module docstring for the individual steps).

    Parameters
    ----------
    study : ChordStudy
        The MSK-CHORD study files.
    config : EntityConfig
        Entity definition (e.g. ``chordutils.entities.PDAC``).
    require_diagnosis : bool, default True
        Remove patients without a matching registry diagnosis from all tables. Without a
        diagnosis there is no time origin for the timeline analyses.
    drop_constant : bool, default False
        Drop columns that are constant within the cohort (see ``drop_constant_columns``).
        Reproduces the original PDAC analysis; the dropped columns are listed in the log.

    Returns
    -------
    Cohort
    """

    log = AttritionLog()

    # 1. samples ---------------------------------------------------------------------------
    samples = study.sample()
    log.note(samples, "sample", "all samples", "data_clinical_sample.txt")
    for col, values in config.sample_filters.items():
        values = list(values)
        shown = ", ".join(map(str, values[:3])) + (f", ... ({len(values)} values)" if len(values) > 3 else "")
        samples = log.filter(samples, samples[col].isin(values), "sample", f"{col} filter", f"{col} in [{shown}]")
    if drop_constant:
        samples = drop_constant_columns(samples, log, "sample")
    patient_ids = samples[ID].unique()

    # 2. sequencing day ----------------------------------------------------------------------
    spec = study.timeline("specimen")
    spec = spec.loc[spec["SAMPLE_ID"].isin(samples["SAMPLE_ID"]), ["SAMPLE_ID", "START_DATE"]]
    spec = spec.rename(columns={"START_DATE": "SEQ_DATE"})
    merged = samples.merge(spec, on="SAMPLE_ID")
    log.record(samples, merged, "sample", "sequencing day", "SEQ_DATE from specimen timeline (inner join)")
    samples = merged

    # 3. patients ----------------------------------------------------------------------------
    patients = study.patient()
    patients = log.filter(patients, patients[ID].isin(patient_ids), "patient", "cohort patients",
                          "at least one selected sample")
    if drop_constant:
        patients = drop_constant_columns(patients, log, "patient")
    if "OS_STATUS" in patients.columns:
        patients = patients.assign(OS_STATUS=parse_os_status(patients["OS_STATUS"]))

    # 4. diagnosis ---------------------------------------------------------------------------
    dx = study.timeline("diagnosis")
    dx = log.filter(dx, dx[ID].isin(patient_ids), "diagnosis", "cohort patients", "at least one selected sample")
    dx = select_diagnoses(dx, config, log).reset_index(drop=True)
    if drop_constant:
        dx = drop_constant_columns(dx, log, "diagnosis")
    if "SUMMARY" in dx.columns:
        dx = dx.assign(SUMMARY=dx["SUMMARY"].str.strip())

    # 5. patients without diagnosis ----------------------------------------------------------
    if require_diagnosis:
        with_dx = dx[ID].unique()
        samples = log.filter(samples, samples[ID].isin(with_dx), "sample", "patients with diagnosis",
                             "time origin (diagnosis) required")
        patients = log.filter(patients, patients[ID].isin(with_dx), "patient", "patients with diagnosis",
                              "time origin (diagnosis) required")

    return Cohort(config=config, samples=samples, patients=patients, diagnosis=dx, log=log)
