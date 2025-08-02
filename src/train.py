import optuna
import pytorch_lightning as pl
import torch as t
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from config import DEVICE, MUTATION_COL, NUMERICAL_COLS, PROTEIN_SELECTED
from src.etl import get_train_test_data, preprocess_fold

from .models import CancerPredictor


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
        mutation_id = self.mutation_to_idx.get(
            mutation_string, self.mutation_to_idx["<UNK>"]
        )

        label = self.df.loc[sample_id]["tumor_type"]
        label = self.label_encoder.transform([label])[0]

        numerical_features = t.tensor(numerical_features.astype(float).values, dtype=t.float32)
        mutation_id = t.tensor(mutation_id, dtype=t.long)
        label = t.tensor(label, dtype=t.long)

        return numerical_features, mutation_id, label


def train_model(train_loader, val_loader, test_loader=None, **kwargs):
    """
    Trains a model and tests it on the test set.
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="./checkpoints",
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=10,
        gradient_clip_val=5,
        callbacks=[checkpoint_callback, learning_rate_callback],
        enable_progress_bar=True,
    )

    model = CancerPredictor(**kwargs)
    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.validate(val_loader, ckpt_path="best", verbose=False)
    if test_loader:
        test_results = trainer.test(test_loader, ckpt_path="best", verbose=False)
        output_test_result = test_results[0]["test_acc"]
    else:
        output_test_result = None

    results = {
        "best_val_loss": checkpoint_callback.best_model_score.item(),
        "val_acc": val_result[0]["val_acc"],
        "test_acc": output_test_result,
    }

    return checkpoint_callback.best_model_path, results


def tune_hyperparameters():
    raw_train_val_df, raw_test_df = get_train_test_data()
    train_df, val_df, trainsforms = preprocess_fold(raw_train_val_df, raw_test_df)
    mutation_to_idx = trainsforms["mutation_to_idx"]
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["tumor_type"])

    train_loader = t.utils.data.DataLoader(
        CancerDataset(
            train_df, mutation_to_idx, NUMERICAL_COLS, MUTATION_COL, label_encoder
        ),
        batch_size=32,
        num_workers = 3,
    )
    val_loader = t.utils.data.DataLoader(
        CancerDataset(
            val_df, mutation_to_idx, NUMERICAL_COLS, MUTATION_COL, label_encoder
        ),
        batch_size=32,
        num_workers = 3,
    )

    def objective(trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "embed_dim": trial.suggest_int("embed_dim", 4, 16, step=4),
            "num_mutation_types": len(mutation_to_idx),
            "dropout_prob": trial.suggest_float("dropout_prob", 0.1, 0.5, step=0.1),
        }
        _, results = train_model(train_loader, val_loader, **params)
        return results["best_val_loss"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("Best hyperparameters found: ", study.best_params)
    return study.best_params, study.best_value


def run_cross_validation():
    pass
