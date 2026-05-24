# src/config.py
# ── Single source of truth for all model hyperparameters ──────────────────────
# If you retrain with different settings, only this file needs to change.

MAX_LENGTH    = 40      # Maximum token sequence length (must match tokenizer)
VOCAB_SIZE    = 10000   # Vocabulary size (must match tokenizer)
EMBEDDING_DIM = 512     # Transformer embedding dimension
UNITS         = 512     # Feed-forward layer units
SEQ_LENGTH    = 25      # Max tokens to generate during inference
IMAGE_SIZE    = (299, 299)  # InceptionV3 input size