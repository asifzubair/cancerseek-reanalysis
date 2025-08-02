import pytorch_lightning as pl
import torch as t
from models import CancerPredictor
from torch.utils.data import Dataset

from config import DEVICE


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

        numerical_features = t.tensor(numerical_features.values, dtype=t.float32)
        mutation_id = t.tensor(mutation_id, dtype=t.long)
        label = t.tensor(label, dtype=t.long)

        return numerical_features, mutation_id, label


def train_model(train_loader, val_loader, test_loader, **kwargs):
    """
    Trains a model and tests it on the test set.
    """
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
    learning_rate_callback = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        accelerator="gpu" if str(DEVICE).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=10,
        gradient_clip_val=5,
        callbacks=[checkpoint_callback, learning_rate_callback],
        enable_progress_bar=True,
    )

    model = CancerPredictor(max_iters=trainer.max_epochs * len(train_loader), **kwargs)
    trainer.fit(model, train_loader, val_loader)

    # The trainer automatically loads the best checkpoint from the .fit() call.
    test_results = trainer.test(dataloaders=test_loader, verbose=False)

    results = {
        "best_val_loss": checkpoint_callback.best_model_score.item(),
        "test_acc": test_results[0]["test_acc"],
    }
    return checkpoint_callback.best_model_path, results
