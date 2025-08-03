import os

import torch as t

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

METADATA_COl_NAMES = [
    "patient_id",
    "sample_id",
    "tumor_type",
    "ajcc_stage",
    "cancerseek_score",
    "cancerseek_result",
]
NON_CANCER_STATUS = "Normal"

## Mutation varaibles
MUTATION_COL_NAMES = [
    "patient_id",
    "sample_id",
    "tumor_type",
    "ajcc_stage",
    "plasma_volume",
    "plasma_dna_concentration",
    "mutation_identified_in_plasma",
    "omega_score",
    "mutant_allele_frequency",
    "mutant_fragments_per_ml_plasma",
    "cancerseek_score",
    "cancerseek_result",
]

MUTATION_FEATURES = [
    "omega_score",
    "mutant_allele_frequency",
    "mutant_fragments_per_ml_plasma",
]

## Protein variables
PROTEIN_COL_NAMES = [
    "patient_id",
    "sample_id",
    "tumor_type",
    "ajcc_stage",
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
    "cancerseek_score",
    "cancerseek_result",
]

PROTEIN_FEATURES = PROTEIN_COL_NAMES[4:-2]

PROTEIN_SELECTED = [
    "ca_125",
    "cea",
    "ca19_9",
    "prolactin",
    "hgf",
    "opn",
    "myeloperoxidase",
    "timp_1",
]

PROTEIN_NORMALIZATION_QUANTILE = 0.95

## Model & Trainer parameters

DEVICE = t.device("cuda" if t.cuda.is_available() else "cpu")

NUMERICAL_COLS = (
    PROTEIN_FEATURES
    + [f"{p}_is_censored" for p in PROTEIN_SELECTED]
    + MUTATION_FEATURES
)

CLASSIFIER_COLS = (
    PROTEIN_SELECTED
    + [f"{p}_is_censored" for p in PROTEIN_SELECTED]
    + MUTATION_FEATURES
)
STANDARDIZE_COLS = PROTEIN_SELECTED + MUTATION_FEATURES

MUTATION_COL = "mutation_identified_in_plasma"
LABEL_COL = "tumor_type"
NONE_TOKEN = "None detected"
MIN_MUTATION_COUNT = 2
BEST_PARAMS = {
    "learning_rate": 0.004627654534332516,
    "embed_dim": 4,
    "dropout_prob": 0.2,
}
