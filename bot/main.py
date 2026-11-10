"""
main.py: the Telegram bot. Wiring only: "when THIS kind of message arrives, do THAT".

THE STORY IN ONE PARAGRAPH
A group wants a civil chat without a human moderator on duty. This file listens to
the group. Every text message goes to the Detector (toxic_detector/predict.py), which
returns the labels that fired. No labels: the bot does nothing. Some labels: it deletes
the message and posts a short public notice. Photos, videos, audio files and voice
messages are not judged by AI at all (the model reads text only); the group's rule
is simply that they are not allowed, so they are always deleted.

READING ORDER
(1) on_text: the only place the AI is used. (2) on_media: the rule-based ban.
(3) build_application: the routing table. (4) main: start-up, run once.

WHAT CHANGED FROM THE FIRST VERSION
- python-telegram-bot 13 -> 22. Version 13 cannot run on current Python (it imports
  `imghdr`, removed in Python 3.13). Version 22 is asynchronous: handlers are
  `async def` and Telegram calls are awaited. The behaviour of the bot is unchanged.
- Decisions come from a list of labels, not from comparing sentences (issue 7).
- One media handler + the MEDIA_RULES table instead of four copies (issue 9).
- Names never crash the handler (issue 8), and failures are logged instead of killing
  the handler silently.

HOW TO RUN (from the repository root)
    export TELEGRAM_TOKEN=...      # from @BotFather; never written in code
    export MODEL_PATH=model        # local model folder or Hugging Face repo id
    python -m bot.main
The bot must be an admin of the group with the right to delete messages.
"""
import asyncio
import logging
import os
from functools import reduce

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from toxic_detector.predict import Detector

from .messages import MEDIA_RULES, display_name, media_kind, media_notice, text_notice

log = logging.getLogger("toxic_bot")


def _sender_name(update: Update) -> str:
    user = update.effective_user        # None for posts made on behalf of a channel
    return display_name(user.first_name, user.last_name) if user else display_name(None)


async def _remove_and_explain(update: Update, context: ContextTypes.DEFAULT_TYPE, notice: str):
    """Delete the message, then post the notice. If deleting fails (most often: the bot
    is not an admin), log it and stay silent: a notice about a message that is still
    there would only confuse the group."""
    message = update.effective_message
    try:
        await message.delete()
    except Exception:
        log.exception("Could not delete message %s in chat %s", message.message_id, message.chat_id)
        return
    await context.bot.send_message(chat_id=message.chat_id, text=notice)


# ---------------------------------------------------------------------------
# 1. Text: the only place where the AI is used
# ---------------------------------------------------------------------------
async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # effective_message also covers EDITED messages (update.message is None for those,
    # which crashed the old bot). So editing a message into an insult is caught too.
    message = update.effective_message
    detector: Detector = context.bot_data["detector"]

    # predict() is ordinary, blocking code that keeps the CPU busy for a moment.
    # asyncio.to_thread runs it on a worker thread so the bot keeps receiving other
    # messages meanwhile, instead of freezing for every prediction.
    try:
        labels = await asyncio.to_thread(detector.predict, message.text)
    except Exception:
        # If the model fails, the safe default is to leave the message alone and log.
        log.exception("Prediction failed; message left in place")
        return

    if labels:                          # [] means approved: do nothing
        log.info("Removing message %s: %s", message.message_id, labels)
        await _remove_and_explain(update, context, text_notice(_sender_name(update), labels))


# ---------------------------------------------------------------------------
# 2. Banned media: a rule, no AI
# ---------------------------------------------------------------------------
async def on_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    kind = media_kind(update.effective_message)
    if kind:
        await _remove_and_explain(update, context, media_notice(_sender_name(update), kind))


# ---------------------------------------------------------------------------
# 3. Routing: which handler gets which message
# ---------------------------------------------------------------------------
# One combined filter built from the MEDIA_RULES keys: "photo" -> filters.PHOTO, ...
# `reduce` folds the list with `|` ("or"): PHOTO | VIDEO | AUDIO | VOICE.
MEDIA_FILTER = reduce(lambda a, b: a | b, (getattr(filters, kind.upper()) for kind in MEDIA_RULES))


def build_application(token: str, detector: Detector) -> Application:
    app = Application.builder().token(token).build()
    # bot_data is a dict shared by all handlers: the natural place for the ONE model
    # object, loaded once at start-up instead of once per message.
    app.bot_data["detector"] = detector
    app.add_handler(MessageHandler(filters.TEXT, on_text))
    app.add_handler(MessageHandler(MEDIA_FILTER, on_media))
    app.add_error_handler(_on_error)
    return app


async def _on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last line of defence: log anything a handler did not catch, keep the bot running."""
    log.error("Unhandled error while processing %s", update, exc_info=context.error)


# ---------------------------------------------------------------------------
# 4. Start-up
# ---------------------------------------------------------------------------
def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        level=logging.INFO)
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("Set the TELEGRAM_TOKEN environment variable (see .env.example).")

    # Loading is slow, so it happens once, before the bot starts listening.
    source = os.environ.get("MODEL_PATH", "model")
    log.info("Loading model from %s ...", source)
    detector = Detector.load(source, token=os.environ.get("HF_TOKEN"))

    # Polling: the bot keeps asking Telegram "anything new?". Simple, and it needs no
    # public server (the alternative, a webhook, has Telegram call us instead).
    build_application(token, detector).run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
