"""
messages.py: everything the bot SAYS, as small pure functions.

WHY A SEPARATE FILE
"Pure" means: data in, text out, no network, no Telegram, no model. That makes these
functions trivial to test (tests/test_smoke.py calls them with plain values), and
it keeps main.py about one thing only: wiring Telegram events to actions.

The three fixes that live here:
- Issue 7: the wording is built HERE, from the list of labels predict() returns.
  The model no longer writes sentences and nobody compares strings to make decisions.
- Issue 8: display_name() works with whatever name parts exist. The old version
  assumed every user has a last name; users without one crashed the handler BEFORE
  their message was deleted, so moderation silently failed for exactly them.
- Issue 9: one dictionary describes all banned media types, instead of four copies of
  the same handler that differed by one word.
"""
from toxic_detector.config import LABEL_DISPLAY

# Media the group does not allow, and what the bot says about each. The keys are the
# attribute names Telegram uses on a message (message.photo, message.video, ...).
# To ban a new type (e.g. "sticker"), add one line here; no new function is needed.
MEDIA_RULES = {
    "photo": "Pictures are not allowed",
    "video": "Videos are not allowed",
    "audio": "Audio files are not allowed",
    "voice": "Voice messages are not allowed",
}


def display_name(first_name: str | None, last_name: str | None = None) -> str:
    """'dana', 'levi' -> 'Dana Levi'; 'dana', None -> 'Dana'; nothing -> 'Someone'.

    Only the first letter is upper-cased. str.capitalize() would also LOWER-case the
    rest, turning 'McDonald' into 'Mcdonald'.
    """
    parts = [p.strip() for p in (first_name, last_name) if p and p.strip()]
    if not parts:   # e.g. a message posted on behalf of a channel has no user name
        return "Someone"
    return " ".join(p[0].upper() + p[1:] for p in parts)


def describe_labels(labels: list[str]) -> str:
    """['obscene'] -> 'obscene language'
    ['obscene', 'hate_speech', 'insult_or_threat'] ->
        'obscene language, hate speech and insult or threat'"""
    words = [LABEL_DISPLAY.get(label, label.replace("_", " ")) for label in labels]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]


def text_notice(name: str, labels: list[str]) -> str:
    """The public note posted after deleting a text message."""
    return f"{name}: message removed ({describe_labels(labels)})"


def media_kind(message) -> str | None:
    """Which banned media type this message carries, or None.
    Works on anything with .photo/.video/... attributes (a Telegram Message, or a
    simple stand-in object in the tests)."""
    for kind in MEDIA_RULES:
        if getattr(message, kind, None):
            return kind
    return None


def media_notice(name: str, kind: str) -> str:
    """The public note posted after deleting a banned media message."""
    return f"{name}: {MEDIA_RULES[kind]}"
