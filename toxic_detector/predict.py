"""
predict.py: from a raw message to a list of labels. This is the model's only "public
face"; the bot never touches the network, tokenizer or thresholds directly.

THE STORY OF ONE CALL
    "You are an IDIOT!!"  -> preprocess()    same cleaning as in training (issue 5)
                          -> tokenizer       truncated to max_length (issue 6)
                          -> ToxicClassifier logits, one per label
                          -> sigmoid         probabilities between 0 and 1
                          -> thresholds      ["insult_or_threat"]

WHAT CHANGED, AND WHY
- Issue 6: the old predict() called the tokenizer with no length limit. A message
  longer than 512 tokens made DistilBERT crash. Now we always truncate to the same
  max_length the model was trained with (read from the saved model, not guessed).
- Issue 7: the old predict() returned a finished English sentence ("This message was
  approved"), and the bot decided by comparing strings. Change one word and moderation
  silently breaks. Now predict() returns DATA: a list of label names, empty when the
  message is fine. The bot decides what to say. The model reports, the bot speaks.
- scores() exposes the probabilities themselves. The bot does not need them, but the
  notebook and the tests do, and they are what you look at when choosing thresholds.
"""
import torch

from .model import ToxicClassifier
from .preprocess import preprocess


class Detector:
    """Load once, predict many times. Loading takes seconds and hundreds of MB of
    memory, so the bot creates ONE Detector at start-up and reuses it for every message."""

    def __init__(self, model: ToxicClassifier, tokenizer, device: str = "cpu"):
        self.model = model.to(device).eval()   # eval(): switch off dropout
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def load(cls, source: str, token: str | None = None, device: str | None = None):
        """`source`: a local model folder or a Hugging Face repo id (see model.py)."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tokenizer = ToxicClassifier.load(source, token=token, device=device)
        return cls(model, tokenizer, device)

    @property
    def labels(self) -> list[str]:
        return self.model.labels

    # inference_mode: no gradients are tracked (we are not learning here), which saves
    # memory and time. It is the stricter, faster sibling of torch.no_grad().
    @torch.inference_mode()
    def scores_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Probabilities for several texts at once: [{label: probability}, ...]."""
        cleaned = [preprocess(t) for t in texts]
        batch = self.tokenizer(
            cleaned,
            padding=True,                         # same length inside the batch
            truncation=True,                      # issue 6: never exceed the limit
            max_length=self.model.max_length,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(batch["input_ids"], batch["attention_mask"])
        probs = torch.sigmoid(logits).cpu().tolist()   # sigmoid: one independent yes/no per label
        return [dict(zip(self.labels, row)) for row in probs]

    def scores(self, text: str) -> dict[str, float]:
        return self.scores_batch([text])[0]

    def predict(self, text: str) -> list[str]:
        """The labels that fire for `text`, in the model's label order. [] = approved."""
        if not preprocess(text):        # empty or whitespace only: nothing to judge
            return []
        probs = self.scores(text)
        return [label for label in self.labels
                if probs[label] >= self.model.thresholds[label]]
