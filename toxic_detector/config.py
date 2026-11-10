"""
config.py: the single source of truth for the project's settings.

WHY THIS FILE EXISTS
In the first version, the same facts lived in several places: the label list in
the bot code and again in the notebook, the hyperparameters in the notebook and
(different ones) in the README. When two copies disagree, nobody knows which is
true. That is exactly what happened with the README (issue 3).

So every fact that more than one file needs lives here, once. Training, prediction
and the bot all import it. When a run finishes, the notebook saves these values next
to the model, so the README can quote the settings of a run that actually happened.

WHAT IS NOT HERE
Secrets (Telegram and Hugging Face tokens) and machine-specific paths. They come from
environment variables at run time, because they differ per person and per machine,
and a token in code is a token on GitHub.
"""
from dataclasses import asdict, dataclass

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Order matters: position i in the model's output vector is LABELS[i].
# The names describe what the Jigsaw annotators actually marked, not what we wish
# they had marked:
#   - Jigsaw "obscene" is vulgar or profane language. It is NOT the same as sexual
#     content, so the old name "nsfw" promised more than the data contains.
#   - Jigsaw "insult" and "threat" are merged into one label (a comment is positive if
#     either is). This is NOT "bullying": bullying is a repeated pattern between
#     people, and a single comment cannot show that.
LABELS = ("obscene", "hate_speech", "insult_or_threat")

# How each label is built from the original Jigsaw columns (used by data.py).
JIGSAW_SOURCE_COLUMNS = {
    "obscene": ["obscene"],
    "hate_speech": ["identity_hate"],
    "insult_or_threat": ["insult", "threat"],  # positive if ANY source column is 1
}

# Wording shown to people in the chat (used by the bot, never by the model).
LABEL_DISPLAY = {
    "obscene": "obscene language",
    "hate_speech": "hate speech",
    "insult_or_threat": "insult or threat",
}

# ---------------------------------------------------------------------------
# Model and text length
# ---------------------------------------------------------------------------
BASE_MODEL = "distilbert-base-uncased"

# One limit for training AND prediction. DistilBERT cannot read more than 512
# tokens; we use 256 because Jigsaw comments are mostly short and every extra token
# costs GPU time. Longer texts are truncated (the end is cut off).
# Because this single value is shared, training and serving can never disagree on it.
MAX_LENGTH = 256

# ---------------------------------------------------------------------------
# Decision thresholds
# ---------------------------------------------------------------------------
# The model outputs one probability per label; a label "fires" when its probability
# is >= its threshold. All are 0.5 for now. They are kept per label because the
# labels have very different base rates, and a better plan (brief, issue 4) is to
# choose each threshold on the validation set. Not done yet; see README.
THRESHOLDS = {label: 0.5 for label in LABELS}


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
# A dataclass groups the values and can be turned into a dict with asdict(), which
# the notebook writes to disk next to the trained model. The README then quotes that
# file, never a number typed by hand.
@dataclass
class TrainConfig:
    seed: int = 42                 # same seed -> same split and same initialisation
    learning_rate: float = 2e-5
    batch_size: int = 32
    epochs: int = 2
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06     # share of steps where the learning rate ramps up
    val_size: float = 0.1          # share of the data held out for validation
    test_size: float = 0.1         # share of the data held out for the final test
    max_length: int = MAX_LENGTH
    base_model: str = BASE_MODEL

    def to_dict(self) -> dict:
        return asdict(self)
