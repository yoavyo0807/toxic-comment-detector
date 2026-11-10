"""
preprocess.py: the one cleaning function, shared by training and prediction.

THE PROBLEM IT SOLVES (issue 5)
The first version cleaned the training text heavily (lowercasing, stopword removal,
dropping numbers and punctuation), but the bot sent the model raw chat messages.
So the model learned from text like "nt think understand came idea bad" and was then
asked about "I don't think you understand". Same words, different "dialect". A model
is only reliable on input that looks like what it was trained on.

THE RULE
There is exactly one function, preprocess(), and BOTH sides call it: data.py before
training, predict.py before every prediction. If the cleaning ever changes, it changes
for both at once, and the model must be retrained.

WHY SO LIGHT
DistilBERT was pretrained on ordinary, full English sentences. Heavy cleaning removes
exactly what it is good at reading:
  - Stopwords carry meaning here. Remove "not" from "you are not stupid" and you have
    flipped the message.
  - Lowercasing is unnecessary: the "uncased" tokenizer already does it.
  - Punctuation and repeated letters ("!!!", "sooo") are signals of tone.
So we only remove things that carry no linguistic signal for these labels: HTML markup,
the specific characters of a URL, and messy whitespace.
"""
import html
import re

# re.compile builds each pattern once, when the module is imported, instead of on
# every call. preprocess() runs for every chat message, so this is the idiomatic choice.
# A tag must start with a letter (or "/" + letter), so "a < b and c > d" survives.
_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")             # <b>, </div>, <br/> ...
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")                          # spaces, tabs, newlines, runs of them


def preprocess(text) -> str:
    """Return a lightly cleaned version of `text`. Never raises on odd input."""
    # Telegram captions and CSV cells can be missing (None or NaN). Treat anything
    # that is not a string as an empty message rather than crashing the bot.
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)      # "&amp;" -> "&", "&quot;" -> '"'
    text = _TAG.sub(" ", text)      # drop markup, keep the words between tags
    # Replace a link with a word instead of deleting it: "check this url" keeps the
    # fact that a link was there (useful later, e.g. for spam) without the noise of
    # a random-looking address.
    text = _URL.sub(" url ", text)
    text = _SPACE.sub(" ", text)    # Wikipedia comments are full of "\n\n"
    return text.strip()
