import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    MUTATION_COL_NAMES,
    NON_CANCER_STATUS,
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
    return df


def load_protein_data(clean=True):
    """load and optionally clean protein data"""

    df = pd.read_excel(
        os.path.join(DATA_DIR, "NIHMS982921-supplement-Tables_S1_to_S11.xlsx"),
        sheet_name="Table S6",
        skiprows=2,
        skipfooter=4,
    )
    df.columns = PROTEIN_COL_NAMES

    if clean:
        df[PROTEIN_SELECTED] = df[PROTEIN_SELECTED].apply(
            lambda col: col.astype(str).str.replace("*", "", regex=False)
        )
        df[PROTEIN_SELECTED] = df[PROTEIN_SELECTED].astype(float)

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

    return train_df_scaled, test_df_scaled


def get_train_test_data(test_size=0.2, random_state=42):
    """Load, split, and process both protein and mutation data."""

    mutation_df = load_mutation_data()
    protein_df = load_protein_data(clean=True)

    merged_df = pd.merge(
        mutation_df,
        protein_df,
        on="sample_id",
        how="inner",
    )

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
