import pytorch_lightning as pl
import torch as t
import torch.nn as nn
import torch.nn.functional as F


class CancerPredictor(pl.LightningModule):
    def __init__(
        self, num_mutation_types, *args, learning_rate=1e-3, embed_dim=8, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self._create_model()

    def _create_model(self):
        self.mutation_embedding = nn.Embedding(
            self.hparams.num_mutation_types, self.hparams.embed_dim
        )
        self.numerical_features = nn.Sequential(
            nn.Linear(in_features=19, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=8),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(
            in_features=8 + self.hparams.embed_dim, out_features=9
        )

    def forward(self, x_numerical, x_mutation):
        x_numerical = self.numerical_features(x_numerical)
        x_mutation = self.mutation_embedding(x_mutation)
        x = t.cat((x_numerical, x_mutation), dim=1)
        return self.output_layer(x)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def _calculate_loss(self, batch, mode="train"):
        """
        Contains the core logic for a single model step and calculates metrics.
        'mode' can be 'train' or 'val'.
        """
        x_numerical, x_mutation, y = batch
        logits = self.forward(x_numerical, x_mutation)
        loss = F.cross_entropy(logits, y)
        preds = t.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{mode}_loss", loss, on_step=False, on_epoch=True)
        self.log(f"{mode}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
