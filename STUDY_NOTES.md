# Study notes: toxic-comment-detector (private, for interview prep; do not publish)

Format per session: 1) what it does, 2) why it was built this way, 3) say it back in your own words, 4) likely interview questions.

## Session plan
1. Big picture: Telegram message -> model -> delete/approve. What each file does.
2. Data: Jigsaw labels, obscene->nsfw, threat+insult->bullying, cleaning steps (and why stopword removal is questionable for BERT).
3. Tokenizer + DistilBERT: [CLS] vector (768 dims), linear head, why DistilBERT.
4. Training: forward, loss (BCEWithLogitsLoss expects logits: sigmoid bug in train/val/test steps), optimizer, AUROC and F1.
5. Prediction + thresholds: predict(), 0.5 cut-off, per-label behaviour.
6. The bot: main.py handlers, loading the model once, secrets via environment variables.

## Known issues to fix (and be able to explain)
- Loss gets sigmoid outputs instead of logits (3 places). Retrain, then update README metrics.
- top_layer_classifier.pkl is missing from the repo: document or retrain/save the head.
- "nsfw" is Jigsaw "obscene"; "bullying" = threat OR insult. Describe accurately.
- README lacks per-label metrics, test-split description and limitations.
- Authorship: say clearly what I did vs. my partner.

## Interview Q&A (fill in after each session)
