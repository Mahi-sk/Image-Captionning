# src/inference.py
# ── Weight loading + prediction logic only — architecture lives in model.py ───

import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from src.config import MAX_LENGTH, EMBEDDING_DIM, UNITS, SEQ_LENGTH, IMAGE_SIZE
from src.model  import CNN_Encoder, TransformerEncoderLayer, TransformerDecoderLayer


class CaptionPredictor:

    def __init__(self, weights_dir: str):
        # 1. Load vocabulary
        vocab_path  = os.path.join(weights_dir, "vocab.pkl")
        with open(vocab_path, "rb") as f:
            self.vocab = [str(v) for v in pickle.load(f)]   # strip np.str_ wrappers

        self.vocab_size = len(self.vocab)
        self.word2idx   = {word: idx for idx, word in enumerate(self.vocab)}

        # 2. Build sub-layers
        self.cnn_model = CNN_Encoder()
        self.encoder   = TransformerEncoderLayer(EMBEDDING_DIM, num_heads=1)
        self.decoder   = TransformerDecoderLayer(
            EMBEDDING_DIM, UNITS, num_heads=8, vocab_size=self.vocab_size)

        # 3. Dummy forward pass — materialises all weight tensors
        self._build_layers()

        # 4. Load saved weights
        # CNN: frozen InceptionV3 — ImageNet weights loaded automatically,
        #      no .h5 restore needed since backbone was never trained.
        enc_path = os.path.join(weights_dir, "encoder_weights.pkl")
        with open(enc_path, "rb") as f:
            self.encoder.set_weights(pickle.load(f))

        dec_path = os.path.join(weights_dir, "decoder_weights.pkl")
        with open(dec_path, "rb") as f:
            self.decoder.set_weights(pickle.load(f))

        print(f"✅ CaptionPredictor ready  |  vocab={self.vocab_size} tokens")

    # ── private ───────────────────────────────────────────────────────────────

    def _build_layers(self):
        """Single dummy pass so every layer allocates its weight tensors."""
        dummy_img   = tf.zeros((1, *IMAGE_SIZE, 3))
        dummy_cap   = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
        img_embed   = self.cnn_model(dummy_img,  training=False)
        enc_out     = self.encoder(img_embed,    training=False)
        dummy_input = dummy_cap[:, :-1]
        dummy_mask  = tf.math.not_equal(dummy_input, 0)
        self.decoder(dummy_input, enc_out, training=False, mask=dummy_mask)

    def _tokenize(self, text: str) -> tf.Tensor:
        """Caption string → padded int32 tensor of shape [1, MAX_LENGTH]."""
        unk     = self.word2idx.get("[UNK]", 1)
        indices = [self.word2idx.get(t, unk) for t in text.lower().split()]
        indices = indices[:MAX_LENGTH]
        indices += [0] * (MAX_LENGTH - len(indices))
        return tf.expand_dims(
            tf.convert_to_tensor(indices, dtype=tf.int32), axis=0)

    def _preprocess_image(self, image_bytes: bytes) -> tf.Tensor:
        """Raw image bytes → normalised [1, H, W, 3] float32 tensor."""
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize(IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        return tf.expand_dims(tf.convert_to_tensor(arr), axis=0)

    # ── public ────────────────────────────────────────────────────────────────

    def predict(self, image_bytes: bytes) -> str:
        """Return a caption string for the supplied raw image bytes."""
        img       = self._preprocess_image(image_bytes)
        img_embed = self.cnn_model(img, training=False)
        enc_out   = self.encoder(img_embed, training=False)

        decoded_caption = "[start]"

        for i in range(SEQ_LENGTH - 1):
            tokenized     = self._tokenize(decoded_caption)[:, :-1]
            mask          = tf.math.not_equal(tokenized, 0)
            preds         = self.decoder(tokenized, enc_out, training=False, mask=mask)
            token_idx     = int(np.argmax(preds[0, i, :]))
            sampled_token = self.vocab[token_idx]

            if sampled_token == "[end]":
                break

            decoded_caption += " " + sampled_token

        return decoded_caption.replace("[start]", "").replace("[end]", "").strip()