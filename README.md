# toxic-comment-detector

A Telegram moderation bot backed by a fine-tuned **DistilBERT** multi-label classifier.
Every text message in the group is scored for three kinds of abusive language. If any
label fires, the bot deletes the message and posts a short public notice. Photos,
videos, audio files and voice messages are removed by a simple rule, without AI.

> **Status (November 2026): rewritten, retraining pending.**
> The code has been rewritten and tested (see [What this rewrite fixed](#what-this-rewrite-fixed)).
> The model has **not yet been retrained** with the new pipeline, so this README reports
> **no results yet**. The [Results](#results) section will be filled only from the
> `metrics.json` of an actual run.

---

## How it works

```
Telegram message
  └─ bot/main.py            text? -> ask the model;  photo/video/audio/voice? -> remove (rule)
       └─ predict.py         preprocess() -> tokenizer (max 256 tokens) -> DistilBERT
            └─ model.py      vector of the first token, [CLS] (768 numbers) -> 1 linear layer -> 3 logits
                 └─ sigmoid -> 3 probabilities -> per-label threshold (0.5) -> list of labels
  └─ bot/messages.py        [] -> do nothing;  ["obscene", ...] -> delete + "Dana: message removed (obscene language)"
```

The same `ToxicClassifier` class and the same `preprocess()` function are used in training
and in the bot, so the model that is tested is the model that is served. The training
notebook checks this on every run.

## Labels

Built from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
annotations. The names describe what annotators marked, no more:

| Label | Built from Jigsaw column(s) | What it means | What it does **not** mean |
|---|---|---|---|
| `obscene` | `obscene` | vulgar or profane language | sexual content (the old name "nsfw" overstated this) |
| `hate_speech` | `identity_hate` | abuse aimed at an identity group | every offensive mention of a group |
| `insult_or_threat` | `insult` OR `threat` | an insult or a threat in this comment | bullying, which is a repeated pattern between people, invisible in a single comment |

Jigsaw's `toxic` and `severe_toxic` columns are not used.

## Repository layout

```
toxic_detector/        shared core, used by training AND the bot
  config.py            labels, base model, max length, thresholds, training settings
  preprocess.py        the one cleaning function (HTML, URLs, whitespace; nothing else)
  model.py             ToxicClassifier = DistilBERT + one linear head; save/load (no pickle)
  predict.py           Detector: raw text -> list of labels
  data.py              Jigsaw train.csv -> 3 labels -> seeded, stratified 80/10/10 split
  training.py          Lightning training loop, loss, per-label metrics
bot/
  messages.py          everything the bot says (pure functions)
  main.py              Telegram wiring
notebooks/
  train_colab.ipynb    train, test, save, verify, (optionally) publish, on a Colab GPU
tests/
  test_smoke.py        fast offline checks of the whole pipeline (tiny random model)
```

## Quick start

**1. Install** (Python 3.10+):
```bash
pip install -r requirements.txt          # the bot
pip install -r requirements-train.txt    # + training and tests
pytest -q                                # ~10 s, offline
```

**2. Train** in Google Colab: open `notebooks/train_colab.ipynb`, switch the runtime to GPU,
and run it top to bottom. You will need Kaggle's `train.csv` from the competition page above
(accept the competition rules first). The data is **not** included in this repository and
must not be committed: Kaggle's rules do not allow redistributing it. Run once with
`QUICK_RUN = True` to check the notebook, then with `False` for a real run. You get a
`model/` folder and `metrics.json`.

**3. Run the bot**:
1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Add the bot to your group as an **admin with permission to delete messages**.
   Without that permission it logs an error and leaves messages in place.
3. Copy `.env.example` to `.env`, fill in `TELEGRAM_TOKEN` and `MODEL_PATH` (the `model/`
   folder, or a Hugging Face repo id), then from the repository root:
```bash
set -a; source .env; set +a
python -m bot.main
```
Tokens are only read from environment variables. Never write one into code or a notebook.

## Results

**No results yet.** Every number in this section will be copied from the `metrics.json`
written by a full (non-quick) run of the notebook, together with that run's settings.

| | AUROC | Precision | Recall | F1 |
|---|---|---|---|---|
| `obscene` | *to verify* | *to verify* | *to verify* | *to verify* |
| `hate_speech` | *to verify* | *to verify* | *to verify* | *to verify* |
| `insult_or_threat` | *to verify* | *to verify* | *to verify* | *to verify* |
| macro average | *to verify* | *to verify* | *to verify* | *to verify* |

When filling this in, also report: the share of positives per label in the test set (from
`metrics.json → data.label_summary`), the thresholds used, the training settings, and the
commit the model was trained from. Precision, recall and F1 depend on the threshold;
AUROC does not.

## What this rewrite fixed

| # | Problem in the first version | Fix |
|---|---|---|
| 1 | The loss received probabilities instead of logits (sigmoid applied twice), so the reported loss was meaningless and the learning signal was distorted | `BCEWithLogitsLoss(logits, labels)`; sigmoid only for metrics and prediction |
| 3 | README hyperparameters did not match the notebook | One `TrainConfig`; the values of each run are saved with the model |
| 5 | Training text was heavily cleaned (lowercase, stopwords removed, ...) but the bot sent raw text | One light `preprocess()` used by both training and prediction |
| 6 | No truncation: messages over 512 tokens crashed prediction | Always truncated to the training length (256 tokens) |
| 7 | The bot decided by comparing the model's sentence with `"This message was approved"` | `predict()` returns a list of labels; the bot builds the wording |
| 8 | Users without a last name crashed the handler before their message was deleted | `display_name()` uses whatever name parts exist |
| 9 | Four near-identical media handlers | One handler and one `MEDIA_RULES` table |
| 10 | No tests | `tests/test_smoke.py`, offline, plus an optional check on the real model |
| 11 | The classifier head (`top_layer_classifier.pkl`) was missing; the repo could not run | The model is saved as one folder of `.safetensors` + JSON, no pickle |

Also: `python-telegram-bot` 13 → 22, because version 13 does not run on Python 3.13.
Edited messages are now checked too.

## Known issues

- **Thresholds are not tuned.** Every label fires at probability 0.5. The labels have very
  different base rates, so 0.5 is probably not the best cut-off for any of them. Planned:
  choose each threshold on the validation set, never on the test set.
- **Evaluation is on Jigsaw's natural label distribution only.** The first version trained
  and tested on a "balanced" sub-sample whose construction is not documented and cannot be
  reproduced. The rewrite trains on the full, unbalanced data. A balanced-vs-natural
  comparison has not been done.
- **The first version's deployed model is not reproduced here.** It is not known whether its
  served classifier was the same object as the trained one (the code that produced
  `top_layer_classifier.pkl` is not in the repository). The rewrite replaces it rather than
  verifying it.
- **`hate_speech` has few examples** (about 1% of Jigsaw comments), so its metrics will be
  the least stable. Read them next to the positive counts.
- **Text only.** Captions on photos and videos are not scored (the media is removed anyway),
  and nothing judges images or audio.
- **One message at a time.** The model sees no conversation context: quotes, replies,
  irony and reclaimed words are judged in isolation.

## Limitations

- **Domain shift.** Jigsaw comments come from English Wikipedia talk pages, and the
  annotations were collected for that setting. A Telegram group writes differently: slang,
  emoji, very short messages, mixed languages. Performance in a real chat will differ from
  the test-set numbers, and it is not measured here.
- **English only.** Messages in other languages (Hebrew, Spanish, ...) are out of scope.
  The model will still return scores for them, and those scores are not meaningful.
- **Label definitions and annotator judgement.** The labels reflect Jigsaw's crowd
  annotators. Models trained on this data are known to over-flag comments that merely
  *mention* identity terms (Jigsaw ran a follow-up competition on exactly this bias). Treat
  `hate_speech` with particular care.
- **Truncation.** Only the first 256 tokens are read. Abuse at the end of a very long
  message can be missed.
- **Automatic deletion has no appeal.** Every mistake removes someone's message in public.
  For a real community, prefer flagging messages to a human moderator, or at least logging
  deletions for review.

## History and credits

This project started as the final project of the Data Science certificate at Bar-Ilan
University, built by **Yoav Yosef** and **[partner's name]**, who wrote most of the commits in
the original repository, including the training and preprocessing notebooks.

The original training notebook is kept for reference in `notebooks/original_2024/`. Note that it
trains `AutoModelForSequenceClassification` (with its own built-in head), while the first bot
loaded `AutoModel` plus a separately pickled linear layer; it is not used by the current code.

The first version reported AUROC 0.9304, F1 0.7930 and loss 0.6311 on its test set. Those
numbers are **not** results of this repository: they came from a balanced sub-sample whose
construction is undocumented, from heavily cleaned text, and from training with the loss bug
described above (which makes the loss value meaningless). They cannot be reproduced or
compared with anything here.

The 2026 rewrite (code structure, fixes, tests, notebook and this README) was done by Yoav
Yosef with the help of an AI coding assistant (Claude).

Licensed under the [MIT License](LICENSE).

Contact: yoav.yosef1@gmail.com
