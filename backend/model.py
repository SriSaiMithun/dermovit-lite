"""
DermoViT-Lite: Hybrid CNN + Vision Transformer model for multi-class
skin lesion classification on the HAM10000 dataset.

Architecture idea
------------------
1. A pretrained CNN backbone (EfficientNetB0, ImageNet weights) acts as a
   local-feature extractor and produces a spatial feature map instead of
   a single pooled vector.
2. That feature map is treated as a grid of "patches" (like ViT patch
   embeddings, but coming from CNN features instead of raw pixels — this
   is the "hybrid" part: it keeps the CNN's strong low-data inductive
   bias while adding a Transformer's global self-attention on top).
3. A small Transformer encoder (multi-head self-attention + MLP blocks)
   models long-range relationships between those patches (e.g. relating
   the lesion border to surrounding skin texture).
4. A classification head (GAP + dense + softmax) outputs probabilities
   over the 7 HAM10000 diagnostic classes.

Classes (HAM10000)
-------------------
akiec - Actinic keratoses / intraepithelial carcinoma
bcc   - Basal cell carcinoma
bkl   - Benign keratosis-like lesions
df    - Dermatofibroma
mel   - Melanoma
nv    - Melanocytic nevi
vasc  - Vascular lesions
"""

import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = 224
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Human-readable labels + a short clinical note, used by the API layer
# when it formats a response for the frontend.
CLASS_INFO = {
    "akiec": {"label": "Actinic Keratoses / Intraepithelial Carcinoma", "risk": "high"},
    "bcc": {"label": "Basal Cell Carcinoma", "risk": "high"},
    "bkl": {"label": "Benign Keratosis-like Lesion", "risk": "low"},
    "df": {"label": "Dermatofibroma", "risk": "low"},
    "mel": {"label": "Melanoma", "risk": "high"},
    "nv": {"label": "Melanocytic Nevus (mole)", "risk": "low"},
    "vasc": {"label": "Vascular Lesion", "risk": "low"},
}


@tf.keras.utils.register_keras_serializable(package="DermoViTLite")
class TransformerEncoderBlock(layers.Layer):
    """A single pre-norm Transformer encoder block."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout_rate = dropout
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = models.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ]
        )

    def call(self, x, training=False):
        x = x + self.attn(self.norm1(x), self.norm1(x), training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout_rate,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="DermoViTLite")
class AddPositionEmbedding(layers.Layer):
    """Learnable positional embedding added to the patch token sequence."""

    def build(self, input_shape):
        num_patches, embed_dim = input_shape[1], input_shape[2]
        self.pos_embed = self.add_weight(
            name="pos_embed",
            shape=(1, num_patches, embed_dim),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embed


@tf.keras.utils.register_keras_serializable(package="DermoViTLite")
class BroadcastClsToken(layers.Layer):
    """Repeats the single [CLS] token across the batch dimension.

    A plain subclassed layer instead of a `Lambda` — Keras 3 refuses to
    deserialize saved models containing raw Python lambdas by default
    (safe-mode), which breaks `tf.keras.models.load_model` after saving.
    """

    def call(self, inputs):
        cls_token, tokens = inputs
        return tf.repeat(cls_token, tf.shape(tokens)[0], axis=0)


@tf.keras.utils.register_keras_serializable(package="DermoViTLite")
class TakeClsToken(layers.Layer):
    """Extracts the [CLS] token (position 0) from the token sequence."""

    def call(self, tokens):
        return tokens[:, 0, :]


def build_model(
    img_size: int = IMG_SIZE,
    num_classes: int = NUM_CLASSES,
    embed_dim: int = 256,
    num_heads: int = 4,
    transformer_layers: int = 4,
    mlp_dim: int = 512,
    dropout: float = 0.2,
    freeze_backbone: bool = True,
) -> tf.keras.Model:
    """Builds the hybrid CNN + ViT classifier.

    Parameters mirror the "AI Model Design" section of the Phase-2 report
    (section 4.5) so the report and the code stay in sync: CNN backbone,
    patch/token count, transformer depth, attention heads, and head
    architecture are all configurable from here.
    """
    inputs = layers.Input(shape=(img_size, img_size, 3), name="lesion_image")

    # 1) Data augmentation (train-time only; identity at inference).
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.1)(x)

    # 2) Preprocessing expected by EfficientNet (scales to backbone's range).
    x = tf.keras.applications.efficientnet.preprocess_input(x)

    # 3) CNN backbone -> spatial feature map (no global pooling yet).
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    backbone.trainable = not freeze_backbone
    feature_map = backbone(x)  # shape: (H', W', C), e.g. (7, 7, 1280)

    # 4) Project CNN channels to the Transformer embedding dim, then
    #    flatten the spatial grid into a sequence of "patch" tokens.
    feature_map = layers.Conv2D(embed_dim, kernel_size=1, padding="same")(feature_map)
    h, w = feature_map.shape[1], feature_map.shape[2]
    tokens = layers.Reshape((h * w, embed_dim))(feature_map)

    # 5) Prepend a learnable [CLS] token used for classification.
    cls_token = layers.Embedding(1, embed_dim)(tf.zeros((1, 1), dtype="int32"))
    cls_token = BroadcastClsToken()([cls_token, tokens])
    tokens = layers.Concatenate(axis=1)([cls_token, tokens])

    # 6) Add positional embeddings, then run through the Transformer encoder.
    tokens = AddPositionEmbedding()(tokens)
    for i in range(transformer_layers):
        tokens = TransformerEncoderBlock(
            embed_dim, num_heads, mlp_dim, dropout, name=f"vit_block_{i}"
        )(tokens)

    tokens = layers.LayerNormalization(epsilon=1e-6)(tokens)
    cls_output = TakeClsToken()(tokens)  # take [CLS] token

    # 7) Classification head.
    x = layers.Dense(256, activation="gelu")(cls_output)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs, outputs, name="DermoViT-Lite")
    return model


if __name__ == "__main__":
    m = build_model()
    m.summary()
