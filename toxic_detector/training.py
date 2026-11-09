"""
training.py: how the model learns, and how we measure it. Used only by the notebook.

THE STORY OF ONE TRAINING STEP
    batch of token ids -> ToxicClassifier -> logits (3 raw scores per comment)
    logits + true labels -> loss (one number: how wrong we are)
    loss -> gradients -> optimizer nudges every weight to be a little less wrong

PyTorch Lightning runs the loop around this (epochs, batches, moving data to the GPU,
logging). We only describe ONE step and what to measure; Lightning repeats it.
The model itself lives in model.py: this file wraps it, it does not redefine it.

THE LOSS BUG, AND THE FIX (issue 1)
BCEWithLogitsLoss means "binary cross-entropy, WITH the sigmoid built in". It expects
raw logits and applies the sigmoid itself, in a numerically safe way. The old code
applied sigmoid first and then passed the probabilities to this loss, so the sigmoid
ran twice. sigmoid(sigmoid(x)) can only fall between 0.5 and 0.73, so the "probability"
the loss saw could never get close to 0 or 1. The loss therefore has a high floor:
at best -ln(0.5) = 0.69 for every label that is 0, and -ln(0.73) = 0.31 for every
label that is 1. That is why the old reported loss (0.63) says nothing about quality,
and why the learning signal was weak and distorted.
Fix: the loss gets logits; sigmoid is applied only for metrics and in predict.py.

WHAT WE MEASURE, AND WHY PER LABEL
A single averaged score hides the weak label. Hate speech is far rarer than obscene
language, so the model is usually worst on it, and the average can look fine anyway.
For each label we report:
  - AUROC: how well the scores rank positives above negatives, at ANY threshold.
  - precision / recall / F1 at the CHOSEN threshold: what the bot would actually do.
    precision = of the messages we would delete, how many deserved it.
    recall    = of the messages that deserved deletion, how many we caught.
Macro averages (plain mean over labels) are logged too, for a one-line summary.
"""
import lightning as L
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (MultilabelAUROC, MultilabelF1Score,
                                         MultilabelPrecision, MultilabelRecall)
from transformers import get_linear_schedule_with_warmup

from .config import TrainConfig
from .model import ToxicClassifier


def _metrics(n_labels: int, prefix: str) -> MetricCollection:
    # average=None -> one value per label instead of a single average.
    return MetricCollection({
        "auroc": MultilabelAUROC(n_labels, average=None),
        "precision": MultilabelPrecision(n_labels, average=None),
        "recall": MultilabelRecall(n_labels, average=None),
        "f1": MultilabelF1Score(n_labels, average=None),
    }, prefix=prefix)


class ToxicTrainer(L.LightningModule):
    def __init__(self, model: ToxicClassifier, cfg: TrainConfig = TrainConfig()):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.loss_fn = nn.BCEWithLogitsLoss()     # expects LOGITS (see header)
        n = len(model.labels)
        # Separate metric objects per stage: each one accumulates over a whole epoch,
        # and validation numbers must never mix with test numbers.
        self.val_metrics = _metrics(n, "val/")
        self.test_metrics = _metrics(n, "test/")
        # Thresholds as a tensor, registered as a buffer so Lightning moves it to the GPU
        # together with the model. persistent=False: not saved inside checkpoints.
        thr = torch.tensor([model.thresholds[l] for l in model.labels])
        self.register_buffer("thresholds", thr, persistent=False)
        self.test_results: dict = {}              # filled at the end of trainer.test()

    # -------------------------------------------------------------- one step per stage
    def _step(self, batch):
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        loss = self.loss_fn(logits, batch["labels"])     # issue 1: logits, not sigmoid(logits)
        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, _ = self._step(batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss                                       # Lightning calls backward() on it

    def _eval_step(self, batch, metrics, stage: str):
        loss, logits = self._step(batch)
        probs = torch.sigmoid(logits)                     # sigmoid only now, for metrics
        labels = batch["labels"].long()
        # AUROC needs the probabilities; the threshold metrics need the decisions the
        # bot would make. Binarising with our own per-label thresholds keeps the two
        # consistent with predict.py (torchmetrics alone allows only one threshold).
        decisions = (probs >= self.thresholds).float()
        metrics["auroc"].update(probs, labels)
        for name in ("precision", "recall", "f1"):
            metrics[name].update(decisions, labels)
        # on_epoch=True: Lightning averages the loss over all batches of the epoch.
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, self.test_metrics, "test")

    # ----------------------------------------------------------------- end of epochs
    def _summarise(self, metrics) -> dict:
        """Compute the epoch's metrics, log per-label and macro values, reset."""
        out = {}
        for key, per_label in metrics.compute().items():  # key e.g. "val/f1"
            for label, value in zip(self.model.labels, per_label):
                out[f"{key}/{label}"] = value.item()
            out[f"{key}/macro"] = per_label.mean().item()
        self.log_dict(out)
        metrics.reset()                                   # start the next epoch from zero
        return out

    def on_validation_epoch_end(self):
        self._summarise(self.val_metrics)

    def on_test_epoch_end(self):
        self.test_results = self._summarise(self.test_metrics)

    # --------------------------------------------------------------------- optimizer
    def configure_optimizers(self):
        # AdamW = Adam with weight decay done correctly (a gentle pull of weights toward
        # zero, against overfitting). By convention, biases and LayerNorm weights are not
        # decayed: they are offsets and scales, and shrinking them does not help.
        no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
        params = [
            {"params": [p for n, p in self.named_parameters() if not n.endswith(no_decay)],
             "weight_decay": self.cfg.weight_decay},
            {"params": [p for n, p in self.named_parameters() if n.endswith(no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(params, lr=self.cfg.learning_rate)
        # Warmup: start the learning rate at 0 and ramp up during the first steps, then
        # decay it linearly to 0. Big early updates can wreck the pretrained weights
        # before the new, random head has learned anything sensible.
        total = self.trainer.estimated_stepping_batches
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.cfg.warmup_ratio * total), total)
        # "interval": "step" -> update the learning rate after every batch, not every epoch.
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
