import pandas as pd
from config import DATA_DIR, PROTEIN_COL_NAMES, PROTEIN_SELECTED
import os
from sklearn.model_selection import train_test_split

def load_protein_data(clean = True):
    """load and optionally clean protein data"""

    df = pd.read_excel(
        os.path.join(DATA_DIR, "NIHMS982921-supplement-Tables_S1_to_S11.xlsx"),
        sheet_name = "Table S6",
        skiprows = 2, skipfooter = 4)
    df.columns = PROTEIN_COL_NAMES

    if clean:
        df[PROTEIN_SELECTED] = df[PROTEIN_SELECTED].apply(
            lambda col: col.astype(str).str.replace('*', '', regex=False))
        df[PROTEIN_SELECTED] = df[PROTEIN_SELECTED].astype(float)

    return df


def normalize_protein_data(df, return_thresholds = False):
    """normalizes the protein data"""
    normal_samples_protien_levels = df[df.tumor_type == "Normal"][PROTEIN_SELECTED]
    thresholds = normal_samples_protien_levels.quantile(0.95)
    df[PROTEIN_SELECTED] = df[PROTEIN_SELECTED].where(df[PROTEIN_SELECTED] >= thresholds, 0)
    if return_thresholds:
        return df, thresholds
    return df


def load_mutation_data():
    pass


def get_train_test_data(df):
    protein_df = load_protein_data()
    pass
