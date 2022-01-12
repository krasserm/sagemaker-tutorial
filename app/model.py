import torch.nn as nn
import torchmetrics as tm
import torchvision.models as tvm
import pytorch_lightning as pl


class ResNet18(pl.LightningModule):
    def __init__(self, num_classes: int):
        super().__init__()
        self.save_hyperparameters()
        self.acc = tm.classification.accuracy.Accuracy()
        self.loss = nn.CrossEntropyLoss()
        self.model = tvm.resnet18(num_classes=num_classes)

    def forward(self, batch):
        x, y = batch
        return self.model(x), y

    def step(self, batch):
        logits, y = self(batch)
        loss = self.loss(logits, y)
        y_pred = logits.argmax(dim=-1)
        acc = self.acc(y_pred, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("train_loss", loss)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self.step(batch)
        self.log("test_loss", loss, sync_dist=True)
        self.log("test_acc", acc)
