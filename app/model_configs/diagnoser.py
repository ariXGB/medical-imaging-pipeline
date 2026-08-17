import lightning as L
import torch
import torch.nn as nn

import torchxrayvision as xrv

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall
)

from torchvision.models import densenet161,DenseNet161_Weights

class ChestClassifier(L.LightningModule):

    def __init__(
        self,
        num_classes,
        lr,
        weight_decay,
        freeze_backbone = False
    ):
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.save_hyperparameters()

        # Backbone
        self.model = densenet161(weights=DenseNet161_Weights.DEFAULT)

        self.model.classifier = nn.LazyLinear(
            out_features=num_classes
        )

        # Freeze backbone check
        if freeze_backbone:
            for param in self.model.features.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True

        # Loss

        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = MulticlassAccuracy(
            num_classes=num_classes
        )

        self.val_acc = MulticlassAccuracy(
            num_classes=num_classes
        )

        self.val_f1 = MulticlassF1Score(
            num_classes=num_classes,
            average="macro"
        )

        self.val_recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        )

    def forward(self, x):
        return self.model(x)

    # TRAINING

    def training_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)

        loss = self.criterion(
            logits,
            y
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        self.train_acc.update(
            preds,
            y
        )

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    # VALIDATION

    def validation_step(self, batch, batch_idx):

        x, y = batch

        logits = self(x)

        loss = self.criterion(
            logits,
            y
        )

        preds = torch.argmax(
            logits,
            dim=1
        )

        self.val_acc.update(
            preds,
            y
        )

        self.val_f1.update(
            preds,
            y
        )

        self.val_recall.update(
            preds,
            y
        )

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "val_f1",
            self.val_f1,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        self.log(
            "val_recall",
            self.val_recall,
            prog_bar=False,
            on_step=False,
            on_epoch=True
        )

    # OPTIMIZER

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            filter(
                lambda p: p.requires_grad,
                self.parameters()
            ),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=2
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_acc",
                "interval": "epoch",
                "frequency": 1
            }
        }

    
    # FINE-TUNING HELPER

    def unfreeze_backbone(self):

        for param in self.model.features.parameters():
            param.requires_grad = True