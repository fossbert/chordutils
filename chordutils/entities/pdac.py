"""Pancreatic ductal adenocarcinoma (PDAC) and its variants.

Reproduces the selection of the original PDAC analysis notebook (database of 11 Apr 2025), with
the corrections listed in CHANGELOG.md (0.1.0).
"""

from ..config import EntityConfig, LineRules, TreatmentRules, icdo_topography_pattern

PDAC = EntityConfig(
    name="PDAC",
    sample_filters={
        # Tumour registry site of the sample
        "DIAGNOSIS_DESCRIPTION": ["Pancreas"],
        # OncoTree: exocrine carcinomas incl. invasive IPMN/ITPN, without neuroendocrine,
        # acinar and solid pseudopapillary tumours
        "CANCER_TYPE_DETAILED": [
            "Pancreatic Adenocarcinoma",
            "Adenosquamous Carcinoma of the Pancreas",
            "Undifferentiated Carcinoma of the Pancreas",
            "Intraductal Papillary Mucinous Neoplasm",
            "Osteoclastic Giant Cell Tumor",
            "Intraductal Tubulopapillary Neoplasm",
        ],
        # ICD-O histology, oriented at the WHO 2019 classification of ductal adenocarcinoma
        # and its variants (colloid, signet ring, adenosquamous, undifferentiated, osteoclast-like
        # giant cells). Spelling errors are those of the source data.
        "ICD_O_HISTOLOGY_DESCRIPTION": [
            "Adenocarcinoma, Nos", "Carcinoma, Nos",
            "Infiltrating Duct Carcinoma",
            "Adenocarcinoma Combined With Other Types Of Carcinoma",
            "Mucinous Adenocarcinoma", "Tubular Adenocarcinoma",
            "Carcinoma With Osteoclast-Like Giant Cells",
            "Adenosquamous Carcinoma",
            "Signet Ring Cel Carcinoma",
            "Infiltrating Duct Mixed W/Other Types Of Ca (C50._)",
            "Cystadenocarcinoma, Nos", "Carcinoma, Undifferentiated Type, Nos",
            "Intraductal Papillary-Mucinous Ca Invasive (C25._)",
            "Mucin-Producing Adenocarinoma",
            "Intraductal Papillary Adenoca W/ Invasion",
        ],
    },
    # ICD-O-3 C25 = pancreas. Selects the same diagnoses as the notebook's
    # DX_DESCRIPTION.contains("PANCREAS") in this cohort (checked against the raw data).
    diagnosis_pattern=icdo_topography_pattern(["C25"]),
    multiple_diagnoses="first",
    windows={
        # Days relative to the diagnosis, both bounds included.
        "surgery": (-90, None),             # staging laparoscopy / biopsy shortly before registry date
        "performance_status": (-90, None),
        "treatment": (0, None),             # therapies before diagnosis were reviewed and belong to other tumours
        "radiation": (0, None),
        # Markers up to 90 days before diagnosis are kept (baseline values). The notebook code
        # (`>= .90`) kept only markers from day 1 after diagnosis, contrary to its stated intent;
        # corrected 2026-09-29 (see CHANGELOG.md).
        "tumor_markers": (-90, None),
    },
    treatment=TreatmentRules(
        subtypes=("Chemo", "Targeted", "Biologic", "Immuno", "Other"),  # no Hormone, Bone Treatment
        agents=(
            "FLUOROURACIL", "LEUCOVORIN", "GEMCITABINE", "OXALIPLATIN", "IRINOTECAN", "PACLITAXEL PROTEIN-BOUND",
            "CAPECITABINE", "INVESTIGATIVE", "IRINOTECAN LIPOSOMAL", "CISPLATIN", "OLAPARIB", "ERLOTINIB",
            "CARBOPLATIN", "PACLITAXEL", "PEMBROLIZUMAB", "DOCETAXEL", "IPILIMUMAB", "NIVOLUMAB", "TRAMETINIB",
            "ENCORAFENIB", "BINIMETINIB", "BEVACIZUMAB", "LAROTRECTINIB", "OSIMERTINIB", "NIRAPARIB",
            "TRASTUZUMAB", "RUCAPARIB", "AFATINIB",
        ),
        min_row_span={"CAPECITABINE": 7},          # capecitabine records spanning < 7 days
        pivot_min_size=14,
        min_episode_days=2,                        # single-day episodes
        min_regimen_days={"CAPECITABINE": 21},     # capecitabine alone for less than one cycle
    ),
    tumor_markers=("CA19-9 (U/mL)", "CEA (ng/mL)"),
    marker_uln={"CA19-9 (U/mL)": 37.0, "CEA (ng/mL)": 5.0},
    # Line rules and protocol names of the original PDAC endpoint analysis
    lines=LineRules(
        grace_days=28,
        gap_days=90,
        ignored_agents=("LEUCOVORIN",),
        equivalent_agents={"CAPECITABINE": "FLUOROPYRIMIDINE", "FLUOROURACIL": "FLUOROPYRIMIDINE"},
        regimen_names=(
            ("FOLFIRINOX", frozenset({"FLUOROURACIL", "IRINOTECAN", "OXALIPLATIN"})),
            ("FOLFIRINOX", frozenset({"CAPECITABINE", "FLUOROURACIL", "IRINOTECAN", "OXALIPLATIN"})),
            ("NALIRIFOX", frozenset({"FLUOROURACIL", "IRINOTECAN LIPOSOMAL", "OXALIPLATIN"})),
            ("NAL-IRI/5-FU", frozenset({"FLUOROURACIL", "IRINOTECAN LIPOSOMAL"})),
            ("NAL-IRI", frozenset({"IRINOTECAN LIPOSOMAL"})),
            ("FOLFOX", frozenset({"FLUOROURACIL", "OXALIPLATIN"})),
            ("CAPOX", frozenset({"CAPECITABINE", "OXALIPLATIN"})),
            ("FOLFIRI", frozenset({"FLUOROURACIL", "IRINOTECAN"})),
            ("5-FU/LV", frozenset({"FLUOROURACIL"})),
            ("CAPECITABINE", frozenset({"CAPECITABINE"})),
            ("GEM/NAB-P", frozenset({"GEMCITABINE", "PACLITAXEL PROTEIN-BOUND"})),
            ("GEM/NAB-P/CIS", frozenset({"GEMCITABINE", "PACLITAXEL PROTEIN-BOUND", "CISPLATIN"})),
            ("GEM/NAB-P/CAP", frozenset({"GEMCITABINE", "PACLITAXEL PROTEIN-BOUND", "CAPECITABINE"})),
            ("GEM/CIS", frozenset({"GEMCITABINE", "CISPLATIN"})),
            ("GEM/CAP", frozenset({"GEMCITABINE", "CAPECITABINE"})),
            ("GEMOX", frozenset({"GEMCITABINE", "OXALIPLATIN"})),
            ("GEM/ERLOTINIB", frozenset({"GEMCITABINE", "ERLOTINIB"})),
            ("GEM", frozenset({"GEMCITABINE"})),
            ("OLAPARIB", frozenset({"OLAPARIB"})),
        ),
    ),
    genes=("KRAS", "TP53", "CDKN2A", "SMAD4", "ARID1A", "KDM6A", "TGFBR2", "RNF43", "GNAS", "BRAF", "ERBB2",
           "MYC", "STK11", "PIK3CA"),
    gene_groups={
        "CORE4": ("KRAS", "TP53", "CDKN2A", "SMAD4"),
        "HRR": ("BRCA1", "BRCA2", "PALB2", "ATM", "CHEK2", "RAD51C", "RAD51D", "BRIP1", "BARD1"),
        "MMR": ("MLH1", "MSH2", "MSH6", "PMS2"),
    },
    # The pancreas itself is reported as 'Other' by the NLP tumour-site model.
    primary_site_categories=("Other",),
    notes="PDAC cohort of the original MSK-CHORD analysis (database of 11 Apr 2025).",
)
