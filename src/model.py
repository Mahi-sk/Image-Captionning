# src/model.py
# ── Architecture definitions — no weight loading, no prediction logic ──────────

import tensorflow as tf
from src.config import MAX_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, UNITS


def CNN_Encoder():
    """Frozen InceptionV3 backbone that outputs a spatial feature map."""
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False,
        weights="imagenet"
    )
    inception_v3.trainable = False
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    return tf.keras.models.Model(inception_v3.input, output)


class Embeddings(tf.keras.layers.Layer):
    """Token + positional embeddings."""

    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings    = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))

    def call(self, input_ids):
        length       = tf.shape(input_ids)[-1]
        position_ids = tf.expand_dims(
            tf.range(start=0, limit=length, delta=1), axis=0)
        return (self.token_embeddings(input_ids) +
                self.position_embeddings(position_ids))


class TransformerEncoderLayer(tf.keras.layers.Layer):
    """Single self-attention encoder block for image features."""

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention    = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense        = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training=False):
        x           = self.layer_norm_1(x)
        x           = self.dense(x)
        attn_output = self.attention(
            query=x, value=x, key=x,
            attention_mask=None, training=training)
        return self.layer_norm_2(x + attn_output)


class TransformerDecoderLayer(tf.keras.layers.Layer):
    """Masked self-attention + cross-attention decoder block for caption generation."""

    def __init__(self, embed_dim, units, num_heads, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embedding   = Embeddings(vocab_size, embed_dim, MAX_LENGTH)
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1)
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()
        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out         = tf.keras.layers.Dense(vocab_size, activation="softmax")
        self.dropout_1   = tf.keras.layers.Dropout(0.3)
        self.dropout_2   = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids, encoder_output, training=False, mask=None):
        embeddings    = self.embedding(input_ids)
        combined_mask = None
        padding_mask  = None

        if mask is not None:
            causal_mask   = self._causal_mask(embeddings)
            padding_mask  = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attn_out_1 = self.attention_1(
            query=embeddings, value=embeddings, key=embeddings,
            attention_mask=combined_mask, training=training)
        out_1 = self.layernorm_1(embeddings + attn_out_1)

        attn_out_2 = self.attention_2(
            query=out_1, value=encoder_output, key=encoder_output,
            attention_mask=padding_mask, training=training)
        out_2 = self.layernorm_2(out_1 + attn_out_2)

        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        return self.out(ffn_out)

    def _causal_mask(self, inputs):
        input_shape     = tf.shape(inputs)
        batch_size, seq = input_shape[0], input_shape[1]
        i    = tf.range(seq)[:, tf.newaxis]
        j    = tf.range(seq)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, seq, seq))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1),
             tf.constant([1, 1], dtype=tf.int32)], axis=0)
        return tf.tile(mask, mult)