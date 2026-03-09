"""
Keras Attention U-Net
======================
Exact replica of the Colab training architecture so that positional
weight loading (load_weights without by_name) works correctly.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def attention_block(x, g, filters):
    """Attention gate: learns to suppress irrelevant skip-connection activations."""
    theta_x = layers.Conv2D(filters, 1)(x)
    phi_g   = layers.Conv2D(filters, 1)(g)
    add     = layers.Activation("relu")(theta_x + phi_g)
    psi     = layers.Conv2D(1, 1, activation="sigmoid")(add)
    return x * psi


def conv_block(x, filters):
    """Two consecutive Conv2D(3×3, relu) layers."""
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    return x


def build_keras_unet(input_shape=(512, 512, 4)):
    """
    Build the Attention U-Net with the identical layer-creation order
    used during training in Colab, ensuring positional weight loading works.
    """
    inputs = layers.Input(input_shape)

    # ── Encoder ──────────────────────────────────────────────────────────────
    c1 = conv_block(inputs, 64);   p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1,    128);   p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2,    256);   p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3,    512)

    # ── Decoder Block 1 (512 → 256) ─────────────────────────────────────────
    g1 = layers.UpSampling2D()(c4)
    a1 = attention_block(c3, g1, 256)
    u1 = layers.Concatenate()([g1, a1]);  c5 = conv_block(u1, 256)

    # ── Decoder Block 2 (256 → 128) ─────────────────────────────────────────
    g2 = layers.UpSampling2D()(c5)
    a2 = attention_block(c2, g2, 128)
    u2 = layers.Concatenate()([g2, a2]);  c6 = conv_block(u2, 128)

    # ── Decoder Block 3 (128 → 64) ──────────────────────────────────────────
    g3 = layers.UpSampling2D()(c6)
    a3 = attention_block(c1, g3, 64)
    u3 = layers.Concatenate()([g3, a3]);  c7 = conv_block(u3, 64)

    # ── Output ───────────────────────────────────────────────────────────────
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c7)

    return models.Model(inputs, outputs)
