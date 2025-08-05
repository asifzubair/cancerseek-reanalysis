import optuna
import pandas as pd
import pytorch_lightning as pl
import torch as t
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import (
    BEST_PARAMS,
    CLASSIFIER_COLS,
    DEVICE,
    MUTATION_COL,
    NON_CANCER_STATUS,
    NONE_TOKEN,
    PROTEIN_FEATURES,
    PROTEIN_SELECTED,
)
from src.etl import get_train_test_data, preprocess_fold

from .models import Autoencoder, CancerPredictor, LogisticRegressionBaseline


class BaselineDataset(Dataset):
    def __init__(self, df, numerical_cols, label_encoder):
        self.df = df
        self.numerical_cols = numerical_cols
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = row[self.numerical_cols]
        label = self.label_encoder.transform([row["tumor_type"]])[0]
        return (
            t.tensor(features.astype(float).values, dtype=t.float32),
            t.tensor(label, dtype=t.long),
        )


class AutoencoderDataset(Dataset):
    def __init__(self, df, protein_features):
        self.df = df
        self.protein_features = protein_features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = row[self.protein_features]
        return t.tensor(features.astype(float).values, dtype=t.float32)


class CancerDataset(Dataset):
    def __init__(
        self, df, mutation_to_idx, numerical_cols, mutation_col, label_encoder
    ):
        self.df = df.set_index("sample_id")

        self.mutation_to_idx = mutation_to_idx
        self.numerical_cols = numerical_cols
        self.mutation_col = mutation_col
        self.label_encoder = label_encoder

        self.sample_ids = self.df.index.unique().tolist()

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]

        numerical_features = self.df.loc[sample_id][self.numerical_cols]
        mutation_string = self.df.loc[sample_id][self.mutation_col]
        if mutation_string == NONE_TOKEN:
            mutation_id = self.mutation_to_idx["<NONE>"]
        else:
            mutation_id = self.mutation_to_idx.get(
                mutation_string, self.mutation_to_idx["<UNK>"]
            )

        label = self.df.loc[sample_id]["tumor_type"]
        label = self.label_encoder.transform([label])[0]

        numerical_features = t.tensor(
            numerical_features.astype(float).values, dtype=t.float32
        )
        mutation_id = t.tensor(mutation_id, dtype=t.long)
        label = t.tensor(label, dtype=t.long)

        return numerical_features, mutation_id, label


def train_ae(healthy_controls_df):
    """Trains an autoencoder on the healthy controls data."""

    autoencoder_dataset = AutoencoderDataset(
        healthy_controls_df, protein_features=PROTEIN_FEATURES
    )
    autoencoder_loader = DataLoader(
        autoencoder_dataset, batch_size=16, shuffle=True, num_workers=2
    )

    autoencoder = Autoencoder(in_out_dim=len(PROTEIN_FEATURES))
    ae_trainer = pl.Trainer(
        accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=20,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )
    ae_trainer.fit(autoencoder, autoencoder_loader)
    autoencoder.eval()
    return autoencoder


def train_model(train_loader, val_loader, test_loader=None, **kwargs):
    """Trains a model and tests it on the test set"""
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        verbose=True,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=100,
        log_every_n_steps=4,
        gradient_clip_val=5,
        callbacks=[checkpoint_callback, learning_rate_callback, early_stop_callback],
        enable_progress_bar=True,
    )

    model = CancerPredictor(**kwargs)
    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(
        dataloaders=val_loader, ckpt_path="best", verbose=False
    )
    if test_loader:
        test_results = trainer.test(
            dataloaders=test_loader, ckpt_path="best", verbose=False
        )
        output_test_result = test_results[0]["test_acc"]
    else:
        output_test_result = None

    results = {
        "best_val_loss": checkpoint_callback.best_model_score.item(),
        "val_acc": val_result[0]["val_acc"],
        "test_acc": output_test_result,
    }

    return trainer, results


def get_reconstruction_error(df, autoencoder, scaler):
    """Calculate reconstruction error for train and val sets"""
    protein_data = df[PROTEIN_FEATURES]
    protein_data_scaled = scaler.transform(protein_data)
    protein_data_tensor = t.tensor(protein_data_scaled, dtype=t.float32).to(DEVICE)
    reconstructed = autoencoder(protein_data_tensor)
    error = t.mean((protein_data_tensor - reconstructed) ** 2, dim=1)
    return error.cpu().detach().numpy()


def tune_hyperparameters():
    """perform hyperparameter tuning"""

    raw_train_df, raw_val_df = get_train_test_data()

    # Fit scaler on healthy controls
    healthy_controls_df = raw_train_df[
        raw_train_df.tumor_type == NON_CANCER_STATUS
    ].copy()
    protein_scaler = StandardScaler()
    protein_scaler.fit(healthy_controls_df[PROTEIN_FEATURES])

    # Scale healthy data and train autoencoder
    healthy_controls_scaled = protein_scaler.transform(
        healthy_controls_df[PROTEIN_FEATURES]
    )
    healthy_controls_scaled_df = pd.DataFrame(
        healthy_controls_scaled, columns=PROTEIN_FEATURES
    )
    autoencoder = train_ae(healthy_controls_scaled_df)
    autoencoder.to(DEVICE)

    raw_train_df["reconstruction_error"] = get_reconstruction_error(
        raw_train_df, autoencoder, protein_scaler
    )
    raw_val_df["reconstruction_error"] = get_reconstruction_error(
        raw_val_df, autoencoder, protein_scaler
    )

    numerical_cols_with_ae = CLASSIFIER_COLS + ["reconstruction_error"]
    train_df, val_df, transforms = preprocess_fold(
        raw_train_df, raw_val_df, numerical_cols=numerical_cols_with_ae
    )
    mutation_to_idx = transforms["mutation_to_idx"]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["tumor_type"])

    train_loader = DataLoader(
        CancerDataset(
            train_df,
            mutation_to_idx,
            numerical_cols_with_ae,
            MUTATION_COL,
            label_encoder,
        ),
        batch_size=32,
        num_workers=3,
        shuffle=True,
    )
    val_loader = DataLoader(
        CancerDataset(
            val_df, mutation_to_idx, numerical_cols_with_ae, MUTATION_COL, label_encoder
        ),
        batch_size=32,
        num_workers=3,
    )

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "embed_dim": trial.suggest_int("embed_dim", 4, 16, step=4),
            "num_mutation_types": len(mutation_to_idx),
            "dropout_prob": trial.suggest_float("dropout_prob", 0.1, 0.5, step=0.1),
            "num_numerical_features": len(numerical_cols_with_ae),
        }
        _, results = train_model(train_loader, val_loader, **params)
        return results["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters found: ", study.best_params)
    return study.best_params, study.best_value


def run_cross_validation():
    """run 10 fold cross validation."""

    train_df, _ = get_train_test_data(train_only=True)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["tumor_type"])

    fold_scores = []
    all_oof_results = []

    for idx, (train_idx, val_idx) in enumerate(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(
            train_df["sample_id"], train_df["tumor_type"]
        )
    ):
        print(f"Fold {idx + 1}/10")
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        healthy_controls_df = train_fold[
            train_fold.tumor_type == NON_CANCER_STATUS
        ].copy()
        ae_scaler = StandardScaler()
        ae_scaler.fit(healthy_controls_df[PROTEIN_FEATURES])

        healthy_controls_scaled = ae_scaler.transform(
            healthy_controls_df[PROTEIN_FEATURES]
        )
        healthy_controls_scaled_df = pd.DataFrame(
            healthy_controls_scaled, columns=PROTEIN_FEATURES
        )
        autoencoder = train_ae(healthy_controls_scaled_df)

        train_fold["reconstruction_error"] = get_reconstruction_error(
            train_fold, autoencoder, ae_scaler
        )
        val_fold["reconstruction_error"] = get_reconstruction_error(
            val_fold, autoencoder, ae_scaler
        )

        numerical_cols_with_ae = CLASSIFIER_COLS + ["reconstruction_error"]
        train_fold, val_fold, transforms = preprocess_fold(
            train_fold, val_fold, numerical_cols=numerical_cols_with_ae
        )
        mutation_to_idx = transforms["mutation_to_idx"]

        train_loader = DataLoader(
            CancerDataset(
                train_fold,
                mutation_to_idx,
                numerical_cols_with_ae,
                MUTATION_COL,
                label_encoder,
            ),
            batch_size=32,
            num_workers=3,
        )
        val_loader = DataLoader(
            CancerDataset(
                val_fold,
                mutation_to_idx,
                numerical_cols_with_ae,
                MUTATION_COL,
                label_encoder,
            ),
            batch_size=32,
            num_workers=3,
        )

        params = BEST_PARAMS.copy()
        params["num_numerical_features"] = len(numerical_cols_with_ae)
        params["num_mutation_types"] = len(mutation_to_idx)
        trainer, results = train_model(train_loader, val_loader, **params)
        fold_scores.append(results["best_val_loss"])

        model = CancerPredictor.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )
        predictions = trainer.predict(model, dataloaders=val_loader)

        oof_preds_logits = t.cat(predictions)
        oof_preds_probs = F.softmax(oof_preds_logits, dim=1).numpy()
        oof_preds_labels_encoded = t.argmax(oof_preds_logits, dim=1).numpy()
        oof_preds_labels = label_encoder.inverse_transform(oof_preds_labels_encoded)

        fold_df = pd.DataFrame(
            {
                "sample_id": val_fold["sample_id"].values,
                "predicted_label": oof_preds_labels,
                "true_label": val_fold["tumor_type"].values,
            }
        )

        class_names = label_encoder.classes_
        prob_cols = [f"prob_{name}".lower() for name in class_names]
        prob_df = pd.DataFrame(oof_preds_probs, columns=prob_cols)

        fold_df = pd.concat([fold_df, prob_df], axis=1)
        all_oof_results.append(fold_df)

    final_df = pd.concat(all_oof_results, ignore_index=True)
    avg_score = sum(fold_scores) / len(fold_scores)

    return avg_score, final_df


def run_baseline_cross_validation():
    """run 10 fold cross validation for the baseline model"""

    train_df, _ = get_train_test_data(train_only=True)
    label_encoder = LabelEncoder()
    train_df["tumor_type_encoded"] = label_encoder.fit_transform(train_df["tumor_type"])

    baseline_features = PROTEIN_SELECTED + ["omega_score"]
    all_oof_results = []
    fold_scores = []

    for idx, (train_idx, val_idx) in enumerate(
        StratifiedKFold(n_splits=10, shuffle=True, random_state=42).split(
            train_df["sample_id"], train_df["tumor_type"]
        )
    ):
        print(f"Fold {idx + 1}/10")
        train_fold = train_df.iloc[train_idx].copy()
        val_fold = train_df.iloc[val_idx].copy()

        train_fold, val_fold, _ = preprocess_fold(
            train_fold, val_fold, numerical_cols=baseline_features
        )

        train_loader = DataLoader(
            BaselineDataset(train_fold, baseline_features, label_encoder),
            batch_size=32,
            num_workers=3,
            shuffle=True,
        )
        val_loader = DataLoader(
            BaselineDataset(val_fold, baseline_features, label_encoder),
            batch_size=32,
            num_workers=3,
        )

        model = LogisticRegressionBaseline(
            num_features=len(baseline_features),
            num_classes=len(label_encoder.classes_),
        )

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath="./checkpoints_baseline",
            filename="best-model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        early_stop_callback = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=False, mode="min"
        )
        trainer = pl.Trainer(
            accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
            devices=1,
            max_epochs=100,
            callbacks=[checkpoint_callback, early_stop_callback],
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)

        val_result = trainer.validate(
            dataloaders=val_loader, ckpt_path="best", verbose=False
        )
        fold_scores.append(val_result[0]["val_loss"])

        predictions = trainer.predict(model, dataloaders=val_loader, ckpt_path="best")
        oof_preds_probs = t.cat(predictions).numpy()
        oof_preds_labels_encoded = oof_preds_probs.argmax(axis=1)
        oof_preds_labels = label_encoder.inverse_transform(oof_preds_labels_encoded)

        fold_df = pd.DataFrame(
            {
                "sample_id": val_fold["sample_id"].values,
                "predicted_label": oof_preds_labels,
                "true_label": val_fold["tumor_type"].values,
            }
        )

        class_names = label_encoder.classes_
        prob_cols = [f"prob_{name}".lower() for name in class_names]
        prob_df = pd.DataFrame(oof_preds_probs, columns=prob_cols)

        fold_df = pd.concat([fold_df, prob_df], axis=1)
        all_oof_results.append(fold_df)

    final_df = pd.concat(all_oof_results, ignore_index=True)
    avg_score = sum(fold_scores) / len(fold_scores)

    return avg_score, final_df
