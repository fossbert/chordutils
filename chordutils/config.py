"""Entity configuration: everything that differs between tumour entities lives here.

The functions in ``chordutils`` are identical for all entities. What makes an analysis
PDAC-, breast- or lung-specific (which samples belong to the cohort, which registry diagnosis
counts, which time windows are used, ...) is collected in one ``EntityConfig`` object. Configs
for individual entities live in ``chordutils.entities``.

Keeping these decisions in one place has two purposes: a new entity only needs a new config,
and every analytic decision is written down once, with its rationale, where it can be reviewed.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

# (lower, upper) bound in days relative to the date of diagnosis, both included; None = open.
Window = Tuple[Optional[int], Optional[int]]


def icdo_topography_pattern(codes: Sequence[str]) -> str:
    """Regular expression matching DX_DESCRIPTION entries by ICD-O-3 topography code.

    DX_DESCRIPTION in ``data_timeline_diagnosis.txt`` has the form
    ``'<HISTOLOGY> | <SITE> (M<morphology> | C<topography>)'``, e.g.
    ``'ADENOCARCINOMA, NOS | PANCREAS, HEAD (M8140/3 | C250)'``. Matching the topography code is
    more robust than matching the site text: 'COLON' e.g. misses 'CECUM' and 'RECTOSIGMOID
    JUNCTION', which are colorectal as well (C18-C20).

    Parameters
    ----------
    codes : sequence of str
        Topography codes or prefixes, e.g. ``['C25']`` (pancreas), ``['C50']`` (breast),
        ``['C18', 'C19', 'C20']`` (colorectum), ``['C34']`` (lung), ``['C61']`` (prostate).

    Examples
    --------
    >>> import re
    >>> bool(re.search(icdo_topography_pattern(["C25"]), "ADENOCARCINOMA, NOS | PANCREAS, HEAD (M8140/3 | C250)"))
    True
    """

    codes = [c.upper().strip() for c in codes]
    for c in codes:
        if not re.fullmatch(r"C\d{1,3}", c):
            raise ValueError(f"Not an ICD-O-3 topography code or prefix: {c!r}")
    alternatives = "|".join(re.escape(c) for c in codes)
    return rf"\|\s*(?:{alternatives})\d*\)\s*$"


@dataclass(frozen=True)
class TreatmentRules:
    """How the treatment timeline is turned into regimen episodes (see ``tables.treatment_episodes``).

    Parameters
    ----------
    subtypes : sequence of str
        Values of SUBTYPE that are analysed (Chemo, Targeted, Biologic, Immuno, Hormone,
        Bone Treatment, Other).
    agents : sequence of str or None
        Agents that are analysed; None keeps all agents of the selected subtypes.
    min_row_span : dict of {agent: days}
        Raw agent rows with ``STOP_DATE - START_DATE < days`` are removed BEFORE building
        episodes. Intended for oral drugs whose very short records are mostly documentation
        artefacts (e.g. a single capecitabine prescription date).
    pivot_min_size : int
        ``min_size`` of ``pivot_therapies``: minimum length of blocks kept when an episode is
        split into several parts.
    min_episode_days : int
        Episodes with fewer treatment days are removed (``2`` removes single-day episodes,
        which mostly arise from overlapping start/stop days of consecutive regimens).
    min_regimen_days : dict of {regimen: days}
        Episodes of exactly this regimen with fewer days are removed (e.g. capecitabine
        monotherapy shorter than one 21-day cycle).
    """

    subtypes: Sequence[str] = ("Chemo", "Targeted", "Biologic", "Immuno", "Hormone", "Bone Treatment", "Other")
    agents: Optional[Sequence[str]] = None
    min_row_span: Dict[str, int] = field(default_factory=dict)
    pivot_min_size: int = 7
    min_episode_days: int = 1
    min_regimen_days: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class LineRules:
    """How regimen episodes are combined into lines of therapy (see ``lines.build_lines_of_therapy``).

    Parameters
    ----------
    grace_days : int
        Agents starting within this many days of the line start belong to the initial regimen
        (e.g. oxaliplatin added in cycle 2).
    gap_days : int
        A treatment-free interval longer than this starts a new line even if the same agents
        are resumed.
    ignored_agents : sequence of str
        Agents that never define or change a line on their own (e.g. leucovorin, a modulator).
    equivalent_agents : dict of {agent: class}
        Agents that are interchangeable for the line decision (e.g. capecitabine and
        5-FU -> 'FLUOROPYRIMIDINE'): switching between them does not start a new line.
    regimen_names : sequence of (name, agent set)
        Protocol names for exact agent sets (without ``ignored_agents``); first match wins.
        Sets without a name are reported as the sorted agents joined by '+'.
    """

    grace_days: int = 28
    gap_days: int = 90
    ignored_agents: Sequence[str] = ()
    equivalent_agents: Dict[str, str] = field(default_factory=dict)
    regimen_names: Sequence[Tuple[str, frozenset]] = ()


@dataclass(frozen=True)
class EntityConfig:
    """Analytic decisions that define one tumour entity in MSK-CHORD.

    Parameters
    ----------
    name : str
        Short label (e.g. 'PDAC'), used in file names and logs.
    sample_filters : dict of {column: list of allowed values}
        Filters on ``data_clinical_sample.txt``, applied in the given order, each logged as a
        separate step. Typical columns: CANCER_TYPE, DIAGNOSIS_DESCRIPTION (tumour registry
        site), CANCER_TYPE_DETAILED (OncoTree), ICD_O_HISTOLOGY_DESCRIPTION.
    diagnosis_pattern : str
        Regular expression that identifies the entity's entry in ``data_timeline_diagnosis.txt``
        (column DX_DESCRIPTION). Patients can have several registry diagnoses (other cancers);
        only matching entries are used as the diagnosis of this entity. See
        ``icdo_topography_pattern``.
    multiple_diagnoses : {'first', 'error'}
        What to do if a patient has more than one matching diagnosis (e.g. bilateral or
        metachronous breast cancer, a second primary lung cancer): 'first' keeps the earliest
        (the date of first diagnosis is the time origin), 'error' stops so the cases can be
        reviewed. The number of affected patients is always logged.
    windows : dict of {timeline: (lower, upper)}
        Which events of each timeline are kept, in days relative to the date of diagnosis (both
        bounds included, None = open). Events far before the diagnosis usually belong to
        another disease (e.g. an earlier cancer).
    treatment : TreatmentRules
        Selection of agents and cleaning rules for building regimen episodes.
    tumor_markers : list of str
        Values of TEST in the tumour-marker timeline that are relevant for the entity.
    marker_uln : dict of {test: value}
        Upper limit of normal per marker test, used to flag elevated baseline values.
    lines : LineRules
        Rules for combining regimen episodes into lines of therapy.
    genes : list of str
        Genes reported individually by ``covariates.genomic_features``.
    gene_groups : dict of {group: list of genes}
        Gene sets summarised as '<GROUP>_ALT' (any alteration) and '<GROUP>_N' (number of
        altered genes), e.g. homologous recombination repair genes.
    primary_site_categories : list of str
        NLP tumour-site categories (data_timeline_tumor_sites.txt: Lung, Bone, Liver, Lymph
        Nodes, Other, ...) that can represent the primary tumour itself and are therefore not
        counted as metastatic sites (e.g. 'Lung' for NSCLC, 'Other' for the pancreas).
    notes : str
        Free text: rationale and sources for the choices above.
    """

    name: str
    sample_filters: Dict[str, Sequence[str]]
    diagnosis_pattern: str
    multiple_diagnoses: str = "first"
    windows: Dict[str, Window] = field(default_factory=dict)
    treatment: TreatmentRules = field(default_factory=TreatmentRules)
    tumor_markers: Sequence[str] = ()
    marker_uln: Dict[str, float] = field(default_factory=dict)
    lines: LineRules = field(default_factory=LineRules)
    genes: Sequence[str] = ()
    gene_groups: Dict[str, Sequence[str]] = field(default_factory=dict)
    primary_site_categories: Sequence[str] = ()
    notes: str = ""

    def __post_init__(self):

        if self.multiple_diagnoses not in ("first", "error"):
            raise ValueError(f"multiple_diagnoses must be 'first' or 'error', got {self.multiple_diagnoses!r}")
        re.compile(self.diagnosis_pattern)  # fail early on an invalid pattern
        for name, (lo, hi) in self.windows.items():
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"Window for {name!r} has lower > upper bound: {(lo, hi)}")

    def window(self, timeline: str) -> Window:
        """Window of a timeline; (None, None) = no restriction if none is configured."""

        return self.windows.get(timeline, (None, None))
