"""chordutils - analysis helpers for the MSK-CHORD database.

Modules
-------
io          reading the study files (``ChordStudy``), data dictionary, pickle helper
config      ``EntityConfig``: everything that differs between tumour entities
entities    ready-made configs (``entities.PDAC``)
cohort      ``select_cohort`` -> ``Cohort`` (samples, patients, diagnosis, filter log)
timeline    ``align_to_diagnosis``: time from diagnosis + window filter for any timeline
attrition   ``AttritionLog``: every filter step with reason and counts
tables      analysis tables of a cohort + ``build_database`` (dict of DataFrames)
lines       ``build_lines_of_therapy``, ``classify_regimen``
endpoints   follow-up, per-day imaging assessments, TTD/TTNT/rwPFS per line, OS from any origin
survival    Kaplan-Meier with delayed entry, time-varying exposures, landmark, Cox screen
covariates  patient/tumour, genomic (panel-aware), treatment history, baselines, marker kinetics
markers     values within lines of therapy (``marker_course``) and tumgr input (``to_tumgr``)
cbioportal  driver (OncoKB) matrices from cBioPortal 'alterations across samples' exports
therapy     ``pivot_therapies``: one row per agent -> non-overlapping regimen episodes
windows     legacy look-ups around an event (``find_stagings``, ``find_ps``, ``find_markers``)

Typical use
-----------
>>> import chordutils as cu
>>> study = cu.ChordStudy("MSK_CHORD_2024")
>>> cohort = cu.select_cohort(study, cu.entities.PDAC)
>>> cohort.log                       # how the cohort was derived
>>> db = cu.build_database(cohort, study)   # dict of all analysis tables
>>> L = cu.lines.build_lines_of_therapy(db["TREATMENT"], cohort.config.lines)
>>> E = cu.endpoints.line_endpoints(L, cu.endpoints.collapse_assessments(db["STAGING"]),
...                                 cu.endpoints.patient_followup(cohort.patients))

The functions of the former single-file module remain available as ``chordutils.<name>``.
"""

from . import entities

__version__ = "0.3.0"
from .attrition import AttritionLog
from .cohort import Cohort, select_cohort
from . import cbioportal, covariates, endpoints, lines, markers, survival
from .config import EntityConfig, LineRules, TreatmentRules, icdo_topography_pattern
from .io import ChordStudy, pickle_transfer, read_cbio_table, read_clinical_dictionary
from .tables import (build_database, imaging_assessments, performance_status, radiation, surgeries,
                     treatment_episodes, tumor_markers)
from .therapy import (ID_SEP, Therapy, arr_is_consecutive, arr_split_consecutive, pivot_therapies,
                      treat_dict_to_frame)
from .timeline import align_to_diagnosis, drop_constant_columns, in_window
from .windows import (determine_offset_combo, find_markers, find_ps, find_stagings, gen_time_range,
                      multiple_replace, validate_event)
