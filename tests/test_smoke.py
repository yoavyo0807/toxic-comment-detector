"""
test_smoke.py: fast checks that every part of the pipeline still fits together.

WHAT "SMOKE TEST" MEANS
Like switching on a repaired machine to see whether smoke comes out: these tests do
not prove the model is GOOD (that is what the notebook's metrics are for). They prove
the code RUNS and keeps its promises: preprocess() is stable, the split is repeatable,
a saved model reloads identically, predict() returns a list and survives long input,
the loss gets logits, and the bot's helpers handle missing names.

WHY A TINY RANDOM MODEL
Real DistilBERT is a 260 MB download. These tests build a miniature DistilBERT (2 layers,
32 numbers per token) with random weights and a 12-word vocabulary. It has the same
SHAPE of plumbing as the real one, so it catches wiring bugs, and the whole file runs in
seconds with no internet. One optional test uses your real trained model, if you point
MODEL_PATH at it.

RUN (from the repository root):  pytest -q
"""
import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

from bot import main as bot_main
from bot.messages import display_name, media_kind, media_notice, text_notice
from toxic_detector.config import JIGSAW_SOURCE_COLUMNS, LABEL_DISPLAY, LABELS, THRESHOLDS
from toxic_detector.data import load_jigsaw, split
from toxic_detector.model import ToxicClassifier
from toxic_detector.predict import Detector
from toxic_detector.preprocess import preprocess

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
         "you", "are", "not", "stupid", "hello", "nice", "!"]


# ------------------------------------------------------------------- shared fixtures
# A fixture is a piece of set-up that pytest builds once and hands to every test that
# names it as an argument. scope="session" -> built once for the whole run.
@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tiny")
    (folder / "vocab.txt").write_text("\n".join(VOCAB))
    tokenizer = DistilBertTokenizerFast(vocab_file=str(folder / "vocab.txt"))
    torch.manual_seed(0)
    encoder = DistilBertModel(DistilBertConfig(
        vocab_size=len(VOCAB), dim=32, hidden_dim=64, n_layers=2, n_heads=2))
    ToxicClassifier(encoder).save(folder / "model", tokenizer)
    return folder / "model"


@pytest.fixture
def detector(tiny_model_dir):
    return Detector.load(str(tiny_model_dir), device="cpu")


# ------------------------------------------------------------------------ config
def test_every_label_is_fully_described():
    for table in (JIGSAW_SOURCE_COLUMNS, LABEL_DISPLAY, THRESHOLDS):
        assert set(table) == set(LABELS)


# -------------------------------------------------------------------- preprocess
@pytest.mark.parametrize("raw, expected", [
    ("You are NOT stupid!!!", "You are NOT stupid!!!"),   # stopwords, case, punctuation kept
    ("<b>Hi</b>&amp; bye", "Hi & bye"),                    # markup and entities removed
    ("see https://x.com/a?b=1 now", "see url now"),        # link -> placeholder word
    ("line1\n\n\tline2  ", "line1 line2"),                 # whitespace normalised
    ("a < b and c > d", "a < b and c > d"),                # not mistaken for a tag
    (None, ""), (float("nan"), ""), ("", ""),              # odd input never crashes
])
def test_preprocess(raw, expected):
    assert preprocess(raw) == expected


def test_preprocess_is_idempotent():
    # Cleaning already-clean text must change nothing; otherwise training data (cleaned
    # once) and a message cleaned twice by mistake would differ.
    text = "<p>Visit www.spam.biz &amp; say hi\n</p>"
    assert preprocess(preprocess(text)) == preprocess(text)


# -------------------------------------------------------------------------- data
@pytest.fixture
def jigsaw_csv(tmp_path):
    """A small fake train.csv with Jigsaw's columns and realistic, rare labels."""
    rng = np.random.default_rng(0)
    n = 2000
    raw = pd.DataFrame({"id": range(n),
                        "comment_text": [f"comment {i}\n\n<i>hi</i>" for i in range(n)]})
    for col, rate in [("toxic", .1), ("severe_toxic", .01), ("obscene", .06),
                      ("threat", .02), ("insult", .05), ("identity_hate", .03)]:
        raw[col] = (rng.random(n) < rate).astype(int)
    raw.loc[0, "comment_text"] = "   "                      # empty after cleaning
    raw.loc[1, ["insult", "threat"]] = [0, 1]               # threat alone
    path = tmp_path / "train.csv"
    raw.to_csv(path, index=False)
    return path


def test_load_jigsaw_builds_our_labels(jigsaw_csv):
    df = load_jigsaw(jigsaw_csv)
    assert list(df.columns) == ["text", *LABELS]
    assert len(df) == 1999                                  # the empty comment is dropped
    assert df.loc[0, "insult_or_threat"] == 1               # threat alone counts
    assert df.loc[0, "text"] == "comment 1 hi"              # preprocess() was applied


def test_split_is_repeatable_and_disjoint(jigsaw_csv):
    df = load_jigsaw(jigsaw_csv)
    train, val, test = split(df)
    again, _, _ = split(df)
    assert train["text"].equals(again["text"])              # same seed -> same split
    assert len(train) + len(val) + len(test) == len(df)
    assert not set(train["text"]) & set(test["text"])       # no leakage into test
    assert not set(val["text"]) & set(test["text"])


# ------------------------------------------------------------------------- model
def test_forward_returns_one_logit_per_label(detector):
    batch = detector.tokenizer(["hello", "you are stupid !"], padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = detector.model(batch["input_ids"], batch["attention_mask"])
    assert logits.shape == (2, len(LABELS))


def test_save_and_load_give_identical_predictions(tiny_model_dir, tmp_path):
    first = Detector.load(str(tiny_model_dir), device="cpu")
    first.model.save(tmp_path / "copy", first.tokenizer)
    second = Detector.load(str(tmp_path / "copy"), device="cpu")
    a, b = first.scores("you are not stupid"), second.scores("you are not stupid")
    assert a == pytest.approx(b)
    assert not list(tmp_path.glob("copy/*.pkl"))            # issue 11: no pickle


def test_loss_takes_logits_not_probabilities():
    # Issue 1. A perfect prediction must give (almost) zero loss. With the old bug
    # (sigmoid applied before BCEWithLogitsLoss) it could never go below about 0.5.
    loss_fn = nn.BCEWithLogitsLoss()
    perfect_logits = torch.tensor([[20., -20., -20.]])
    target = torch.tensor([[1., 0., 0.]])
    assert loss_fn(perfect_logits, target).item() < 1e-6
    assert loss_fn(torch.sigmoid(perfect_logits), target).item() > 0.5


# ------------------------------------------------------------------------ predict
def test_predict_returns_a_list_of_known_labels(detector):
    out = detector.predict("you are stupid !")
    assert isinstance(out, list) and set(out) <= set(LABELS)   # issue 7: data, not a sentence


def test_predict_on_empty_message_is_approved(detector):
    assert detector.predict("   ") == [] and detector.predict(None) == []


def test_very_long_message_does_not_crash(detector):
    # Issue 6: the old predict() crashed past DistilBERT's 512-token limit.
    assert isinstance(detector.predict("you are stupid " * 3000), list)


def test_thresholds_decide(detector):
    detector.model.thresholds = {label: 0.0 for label in LABELS}   # everything fires
    assert detector.predict("hello") == list(LABELS)
    detector.model.thresholds = {label: 1.01 for label in LABELS}  # nothing can fire
    assert detector.predict("hello") == []


# -------------------------------------------------------------------- bot helpers
def test_display_name_never_crashes():
    # Issue 8: users without a last name crashed the old bot.
    assert display_name("dana", "levi") == "Dana Levi"
    assert display_name("dana", None) == "Dana"
    assert display_name(None, None) == "Someone"
    assert display_name("McDonald") == "McDonald"


def test_notices():
    assert text_notice("Dana", ["obscene", "insult_or_threat"]) == \
        "Dana: message removed (obscene language and insult or threat)"
    assert media_notice("Dana", "voice") == "Dana: Voice messages are not allowed"
    assert media_kind(SimpleNamespace(photo=[1], video=None)) == "photo"
    assert media_kind(SimpleNamespace(text="hi")) is None


# ------------------------------------------------------------ bot handlers (no network)
class FakeMessage:
    """Stands in for a Telegram Message: only what the handlers use."""
    def __init__(self, text=None, **media):
        self.text, self.message_id, self.chat_id, self.deleted = text, 1, 42, False
        for kind in ("photo", "video", "audio", "voice"):
            setattr(self, kind, media.get(kind))

    async def delete(self):
        self.deleted = True


def _run(handler, message, predict=lambda text: []):
    sent = []

    async def send_message(chat_id, text):
        sent.append(text)

    context = SimpleNamespace(bot=SimpleNamespace(send_message=send_message),
                              bot_data={"detector": SimpleNamespace(predict=predict)})
    update = SimpleNamespace(effective_message=message,
                             effective_user=SimpleNamespace(first_name="dana", last_name=None))
    asyncio.run(handler(update, context))
    return message.deleted, sent


def test_clean_text_is_left_alone():
    assert _run(bot_main.on_text, FakeMessage("hello")) == (False, [])


def test_toxic_text_is_removed_with_notice():
    deleted, sent = _run(bot_main.on_text, FakeMessage("x"), predict=lambda t: ["hate_speech"])
    assert deleted and sent == ["Dana: message removed (hate speech)"]


def test_media_is_removed_with_notice():
    deleted, sent = _run(bot_main.on_media, FakeMessage(photo=[1]))
    assert deleted and sent == ["Dana: Pictures are not allowed"]


# ------------------------------------------------- optional: your REAL trained model
# Skipped unless MODEL_PATH points to a model folder (e.g. after training in Colab:
#   MODEL_PATH=model pytest -q). These are sanity checks on extreme, obvious cases,
# not a measure of quality.
@pytest.mark.skipif(not os.path.isdir(os.environ.get("MODEL_PATH", "")),
                    reason="set MODEL_PATH to a trained model folder to run")
def test_real_model_on_obvious_examples():
    real = Detector.load(os.environ["MODEL_PATH"])
    assert real.predict("Have a nice day, thanks for the help!") == []
    assert "insult_or_threat" in real.predict("You are a stupid idiot and I will hurt you")
