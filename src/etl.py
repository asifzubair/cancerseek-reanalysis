import os
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    MIN_MUTATION_COUNT,
    MUTATION_COL,
    MUTATION_COL_NAMES,
    MUTATION_FEATURES,
    NON_CANCER_STATUS,
    NONE_TOKEN,
    NUMERICAL_COLS,
    PROTEIN_COL_NAMES,
    PROTEIN_NORMALIZATION_QUANTILE,
    PROTEIN_SELECTED,
)


def load_mutation_data():
    df = pd.read_excel(
        os.path.join(DATA_DIR, "NIHMS982921-supplement-Tables_S1_to_S11.xlsx"),
        sheet_name="Table S5",
        skiprows=2,
        skipfooter=12,
    )
    df.columns = MUTATION_COL_NAMES
    for col in MUTATION_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[MUTATION_FEATURES] = df[MUTATION_FEATURES].fillna(0)

    return df


def load_protein_data(
    clean=True,
    protein_cols=PROTEIN_COL_NAMES,
    protein_selected=PROTEIN_SELECTED,
):
    """load and optionally clean protein data"""

    df = pd.read_excel(
        os.path.join(DATA_DIR, "NIHMS982921-supplement-Tables_S1_to_S11.xlsx"),
        sheet_name="Table S6",
        skiprows=2,
        skipfooter=4,
    )
    df.columns = protein_cols

    if clean:
        df[[f"{p}_is_censored" for p in protein_selected]] = df[protein_selected].apply(
            lambda col: col.astype(str).str.contains(r"\*", na=False).astype(int)
        )
        df[protein_selected] = df[protein_selected].apply(
            lambda col: col.astype(str).str.replace("*", "", regex=False).astype(float)
        )

    return df


def calculate_thresholds(df, protein_cols):
    """calculates normalization thresholds from the healthy controls"""
    healthy_controls = df[df.tumor_type == NON_CANCER_STATUS][protein_cols]
    return healthy_controls.quantile(PROTEIN_NORMALIZATION_QUANTILE)


def apply_normalization(df, thresholds, protein_cols):
    """applies pre-calculated thresholds"""
    df_processed = df.copy()
    df_processed[protein_cols] = df_processed[protein_cols].where(
        df_processed[protein_cols] >= thresholds, 0
    )
    return df_processed


def fit_and_apply_scaler(train_df, test_df, numerical_cols):
    """fit a `StandardScaler` on the training data and apply it to both train and test data.

    we are being non-rigourous about `test` here .. could be val too

    """
    scaler = StandardScaler()

    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()

    # Fit AND transform the training data
    train_df_scaled[numerical_cols] = scaler.fit_transform(
        train_df_scaled[numerical_cols]
    )

    # ONLY transform the test data
    test_df_scaled[numerical_cols] = scaler.transform(test_df_scaled[numerical_cols])

    return train_df_scaled, test_df_scaled, scaler


def get_train_test_data(test_size=0.2, random_state=42, train_only=False):
    """Load, split, and process both protein and mutation data."""

    mutation_df = load_mutation_data()
    protein_df = load_protein_data(clean=True)

    merged_df = pd.merge(
        mutation_df,
        protein_df,
        on=["sample_id", "tumor_type", "ajcc_stage"],
        how="inner",
    )
    merged_df[NUMERICAL_COLS] = merged_df[NUMERICAL_COLS].fillna(0)

    if train_only:
        return merged_df, None

    patient_labels = merged_df[["sample_id", "tumor_type"]].drop_duplicates()
    train_ids, test_ids = train_test_split(
        patient_labels["sample_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels["tumor_type"],
    )

    train_merged_df = merged_df[merged_df["sample_id"].isin(train_ids)].copy()
    test_merged_df = merged_df[merged_df["sample_id"].isin(test_ids)].copy()

    return train_merged_df, test_merged_df


def preprocess_fold(
    train_df,
    val_df,
    mutation_col=MUTATION_COL,
    numerical_cols=NUMERICAL_COLS,
    protein_cols=PROTEIN_SELECTED,
):
    """
    Performs all 'fit' and 'transform' operations for a single CV fold.
    """

    counts = Counter(train_df[mutation_col])
    vocab = [
        mut
        for mut, count in counts.items()
        if count >= MIN_MUTATION_COUNT and mut != NONE_TOKEN
    ]
    vocab.insert(0, "<UNK>")
    vocab.insert(0, "<NONE>")
    mutation_to_idx = {token: i for i, token in enumerate(vocab)}

    fold_thresholds = calculate_thresholds(train_df, protein_cols)
    train_df = apply_normalization(train_df, fold_thresholds, protein_cols)
    val_df = apply_normalization(val_df, fold_thresholds, protein_cols)

    train_fold, val_fold, scaler = fit_and_apply_scaler(
        train_df, val_df, numerical_cols
    )

    transforms = {
        "mutation_to_idx": mutation_to_idx,
        "fold_thresholds": fold_thresholds,
        "scaler": scaler,
    }

    return train_fold, val_fold, transforms
