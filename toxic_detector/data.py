"""
data.py: from Kaggle's Jigsaw train.csv to three reproducible splits, and from rows
to padded batches. Used only for training; the bot never imports this file.

THE STORY
1. load_jigsaw(): read the original CSV, build our 3 labels from Jigsaw's 6 columns,
   apply the shared preprocess(), drop comments that end up empty.
2. split(): train / validation / test, stratified and seeded, so the same seed always
   gives the same three sets and every label appears in each set at the same rate.
3. CommentDataset + Collator: hand the model one batch at a time, tokenized on the fly.

WHY THE ORIGINAL CSV (and not a cleaned or balanced copy)
The first version trained on files that had been cleaned and "balanced" by steps that
are not in the repo, so nobody (us included) can rebuild them exactly. Starting from
the untouched Kaggle file and doing every step in code makes the run repeatable.
The CSV itself is never committed: Kaggle's rules do not allow redistributing it.

WHY STRATIFY BY LABEL COMBINATION
Toxic labels are rare (most comments have none), and they co-occur (an insult is
often also obscene). A purely random split could, by bad luck, put very few
hate-speech examples in the test set. We treat each combination of labels
(e.g. "obscene + insult_or_threat") as a group and keep each group's share the same in
all three splits. Your partner's notebook used iterative stratification
(scikit-multilearn) for the same goal; the combination approach uses plain
scikit-learn and accepts a seed. In the full data every combination has dozens of rows
or more; very rare combinations (in a small sample) are pooled, see _combination().
"""
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from .config import JIGSAW_SOURCE_COLUMNS, LABELS, MAX_LENGTH, TrainConfig
from .preprocess import preprocess

TEXT_COLUMN = "comment_text"   # name of the text column in Kaggle's train.csv


def load_jigsaw(csv_path) -> pd.DataFrame:
    """Return a DataFrame with a 'text' column and one 0/1 column per label."""
    raw = pd.read_csv(csv_path)
    df = pd.DataFrame({"text": raw[TEXT_COLUMN].map(preprocess)})
    for label, sources in JIGSAW_SOURCE_COLUMNS.items():
        # .max(axis=1) across the source columns = "1 if ANY of them is 1".
        # (The old notebook added threat + insult and then clipped 2 to 1: same result.)
        df[label] = raw[sources].max(axis=1).astype(int)
    return df[df["text"] != ""].reset_index(drop=True)


def _combination(df: pd.DataFrame, min_count: int = 10) -> pd.Series:
    """One string per row naming its label combination, e.g. '101'. Used to stratify.
    Stratification needs several rows per group, so combinations rarer than
    `min_count` (possible in a small quick-run sample) are pooled into one 'rare' group."""
    combo = df[list(LABELS)].astype(str).agg("".join, axis=1)
    counts = combo.map(combo.value_counts())
    combo = combo.where(counts >= min_count, "rare")
    # If even the pooled group is tiny, fold it into the biggest group (usually '000').
    if (combo == "rare").sum() < min_count:
        combo = combo.replace("rare", combo.value_counts().idxmax())
    return combo


def split(df: pd.DataFrame, cfg: TrainConfig = TrainConfig()):
    """Return (train, val, test). Two cuts: first hold out the test set, then cut the
    validation set from what remains, sized so the final shares match the config."""
    rest, test = train_test_split(
        df, test_size=cfg.test_size, stratify=_combination(df), random_state=cfg.seed)
    val_share_of_rest = cfg.val_size / (1 - cfg.test_size)   # 0.1 of all = 0.111 of the rest
    train, val = train_test_split(
        rest, test_size=val_share_of_rest, stratify=_combination(rest), random_state=cfg.seed)
    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True))


def label_summary(**splits: pd.DataFrame) -> pd.DataFrame:
    """Rows, and the share of positives per label, for each split. This table goes into
    the README as-is, because it says how rare each label is, which you need in order
    to read any metric honestly."""
    rows = {name: {"rows": len(part), **part[list(LABELS)].mean().round(4).to_dict()}
            for name, part in splits.items()}
    return pd.DataFrame(rows).T.astype({"rows": int})


class CommentDataset(Dataset):
    """Holds texts and label vectors; item i is (text, tensor of 0./1. per label)."""

    def __init__(self, df: pd.DataFrame):
        self.texts = df["text"].tolist()
        # float32 because the loss compares these targets with the model's float scores.
        self.labels = torch.tensor(df[list(LABELS)].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.labels[i]


class Collator:
    """Turns a list of (text, labels) pairs into one batch of tensors.

    Tokenizing here, per batch, instead of tokenizing the whole dataset in advance
    allows dynamic padding: each batch is padded only to its own longest comment, not to
    max_length. Most comments are short, so most batches are much smaller and faster.
    """

    def __init__(self, tokenizer, max_length: int = MAX_LENGTH):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        texts, labels = zip(*batch)          # [(t1, l1), (t2, l2)] -> (t1, t2), (l1, l2)
        enc = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": torch.stack(labels)}
