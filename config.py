import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROTEIN_COL_NAMES = ["patient_id", "sample_id", "tumor_type", "ajcc_stage",
    "afp",
    "angiopoietin_2",
    "axl",
    "ca_125",
    "ca_15_3",
    "ca19_9",
    "cd44",
    "cea",
    "cyfra_21_1",
    "dkk1",
    "endoglin",
    "fgf2",
    "follistatin",
    "galectin_3",
    "g_csf",
    "gdf15",
    "he4",
    "hgf",
    "il_6",
    "il_8",
    "kallikrein_6",
    "leptin",
    "mesothelin",
    "midkine",
    "myeloperoxidase",
    "nse",
    "opg",
    "opn",
    "par",
    "prolactin",
    "segfr",
    "sfas",
    "shbg",
    "sher2_segfr2_serbb2",
    "specam_1",
    "tgfa",
    "thrombospondin_2",
    "timp_1",
    "timp_2",
    "cancerseek_score", "cancerseek_result"
]

PROTEIN_METADATA = ["patient_id", "sample_id", "tumor_type", "ajcc_stage","cancerseek_score", "cancerseek_result"]
PROTEIN_SELECTED = ["ca_125", "cea", "ca19_9", "prolactin", "hgf", "opn", "myeloperoxidase", "timp_1"]
