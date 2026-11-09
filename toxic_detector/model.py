"""
model.py: the network itself, and how it is saved and loaded.

THE STORY
A comment goes in as token ids. DistilBERT reads the whole sequence and gives back
one 768-number vector per token. We keep only the vector of the first token, [CLS],
as a summary of the whole comment, and pass it through ONE linear layer that turns
768 numbers into 3 scores, one per label. Those raw scores are called logits.
Turning logits into probabilities (sigmoid) and probabilities into decisions
(thresholds) happens elsewhere: in training.py for metrics and in predict.py
for the bot.
This is the same architecture your partner trained ("only one layer"); the rewrite keeps
it so the old experiments stay comparable.

WHY ONE CLASS FOR TRAINING AND SERVING (the root of issues 2 and 11)
In the first version, the notebook built and trained one object, but the bot rebuilt
a DIFFERENT one by hand: a model downloaded from Hugging Face plus a linear layer read
from a pickle file that was never committed. Nothing guaranteed the two matched, and
the repo could not run. Here, ToxicClassifier is the only definition. Training creates
it, save() writes it, load() rebuilds exactly it, and predict.py uses what load() gives.

WHY NOT PICKLE
A pickle stores Python objects together with the code needed to rebuild them, so it
breaks when library versions change, and unpickling a file can run arbitrary code.
We save plain tensors instead: the encoder with Hugging Face's save_pretrained(), the
head as a .safetensors file (a format that only holds numbers), and the facts the
model depends on (label order, max length, thresholds) as readable JSON.
"""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .config import BASE_MODEL, LABELS, MAX_LENGTH, THRESHOLDS

HEAD_FILE = "classifier_head.safetensors"
META_FILE = "detector_config.json"


class ToxicClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, n_labels: int = len(LABELS), dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        # Dropout randomly zeroes 10% of the [CLS] numbers during training only, so the
        # head cannot lean on a few of them. It does nothing in eval mode (prediction).
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(encoder.config.hidden_size, n_labels)

        # Facts that travel with the model. load() fills them from the saved file, so the
        # bot reads the label order the model was trained with, not a separate copy of it.
        self.labels = list(LABELS)
        self.max_length = MAX_LENGTH
        self.thresholds = dict(THRESHOLDS)

    @classmethod
    def from_base(cls, base_model: str = BASE_MODEL, n_labels: int = len(LABELS)):
        """A fresh model for training: pretrained DistilBERT + a new, random head."""
        return cls(AutoModel.from_pretrained(base_model), n_labels=n_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Token ids -> logits with shape (batch, n_labels). No sigmoid here, on purpose:
        the loss function expects logits (see training.py, issue 1)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls_vector = hidden[:, 0]            # (batch, 768): the first token of every comment
        return self.head(self.dropout(cls_vector))

    # ------------------------------------------------------------------ saving / loading
    def save(self, directory, tokenizer, extra: dict | None = None) -> None:
        """Write everything needed to rebuild this exact model into one folder.
        `extra` (e.g. the training config and metrics) is stored for the record."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(directory)
        tokenizer.save_pretrained(directory)       # the tokenizer is part of the model
        # .contiguous() because safetensors refuses tensors that are views of others.
        save_file({k: v.contiguous() for k, v in self.head.state_dict().items()},
                  str(directory / HEAD_FILE))
        meta = {
            "labels": self.labels,
            "max_length": self.max_length,
            "thresholds": self.thresholds,
            **(extra or {}),
        }
        (directory / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, source: str, token: str | None = None, device: str = "cpu"):
        """Rebuild a saved model. `source` is a local folder OR a Hugging Face repo id.
        Returns (model, tokenizer), with the model in eval mode."""
        directory = _resolve(source, token)
        meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))

        model = cls(AutoModel.from_pretrained(directory), n_labels=len(meta["labels"]))
        model.head.load_state_dict(load_file(str(directory / HEAD_FILE)))
        model.labels = meta["labels"]
        model.max_length = meta["max_length"]
        model.thresholds = meta["thresholds"]

        tokenizer = AutoTokenizer.from_pretrained(directory)
        return model.to(device).eval(), tokenizer


def _resolve(source: str, token: str | None) -> Path:
    """Local folder -> use it. Otherwise treat `source` as a Hugging Face repo id and
    download the whole repo once (later calls reuse the local cache)."""
    path = Path(source)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download   # only needed for the Hub case
    return Path(snapshot_download(repo_id=source, token=token))
