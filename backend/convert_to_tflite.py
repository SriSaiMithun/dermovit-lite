"""
Converts the trained dermovit_lite.keras model into a lightweight
dermovit_lite.tflite file for low-memory deployment (Render's free tier
crashed running full TensorFlow inference at ~512MB RAM).

Run once, locally or on Colab, after training:
    python convert_to_tflite.py

The exported model has TWO outputs:
    - class probabilities        shape (1, 7)
    - CLS feature embedding      shape (1, 256)
The embedding is used by the feature-distance out-of-distribution gate
(ood.py / fit_ood_detector.py) to reject images that don't look like the
training data. The probabilities alone cannot do this: the softmax is
overconfident on non-skin images.

How it works: it rebuilds the inference graph by reusing the ORIGINAL trained
model's layer objects directly (same weights, no copying), skipping only the 4
training-only augmentation layers (RandomFlip/Rotation/Zoom/Contrast). Those
are identity at inference, but leaving them in forces the TFLite converter to
pull in heavy "Flex" TF ops. The script verifies BOTH outputs match the
original model before writing artifacts/dermovit_lite.tflite.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import model as model_module  # noqa: F401  registers custom layers - must import before load_model

TRAINED_MODEL_PATH = os.path.join("artifacts", "dermovit_lite.keras")
TFLITE_OUTPUT_PATH = os.path.join("artifacts", "dermovit_lite.tflite")
IMG_SIZE = 224
NUM_CLASSES = 7


def build_inference_model_from_trained(trained: tf.keras.Model) -> tf.keras.Model:
    """Rewires the trained model's layers into a fresh graph that skips the
    augmentation layers and exposes [probabilities, cls_embedding]."""
    skip_names = {"lesion_image", "random_flip", "random_rotation", "random_zoom", "random_contrast"}
    kept_layers = [l for l in trained.layers if l.name not in skip_names]

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="lesion_image_infer")
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    idx = 0
    x = kept_layers[idx](x); idx += 1          # efficientnetb0
    x = kept_layers[idx](x); idx += 1          # conv2d
    tokens = kept_layers[idx](x); idx += 1     # reshape

    broadcast_layer = kept_layers[idx]; idx += 1   # broadcast_cls_token
    concat_layer = kept_layers[idx]; idx += 1      # concatenate
    # The Embedding that produced the raw cls token only took a constant as
    # input, so Keras froze its output as a literal at save time (it is not in
    # trained.layers). Pull the frozen value out of the layer's call node.
    node = broadcast_layer._inbound_nodes[0]
    cls_token_const = tf.constant(node.arguments.args[0][0])

    cls_token = broadcast_layer([cls_token_const, tokens])
    tokens = concat_layer([cls_token, tokens])

    embedding = None
    while idx < len(kept_layers):
        layer = kept_layers[idx]
        tokens = layer(tokens)
        if layer.name.startswith("take_cls_token"):
            embedding = tokens          # (None, 256) CLS feature, post final LayerNorm
        idx += 1

    if embedding is None:
        raise RuntimeError("Could not find the take_cls_token layer to expose the embedding.")
    return models.Model(inputs, [tokens, embedding], name="DermoViT-Lite-Inference")


def _split_outputs(details, arrays):
    """Identify probability vs embedding outputs by their last dimension."""
    probs = emb = None
    for d, a in zip(details, arrays):
        if d["shape"][-1] == NUM_CLASSES:
            probs = a
        else:
            emb = a
    return probs, emb


def main():
    print(f"Loading trained model from {TRAINED_MODEL_PATH} ...")
    trained = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)

    print("Rebuilding inference-only graph (reusing trained layers, skipping augmentation) ...")
    inference_model = build_inference_model_from_trained(trained)

    # Reference outputs from the ORIGINAL trained model.
    cls_extractor = models.Model(trained.inputs, trained.get_layer("take_cls_token").output)
    test_input = np.random.rand(3, IMG_SIZE, IMG_SIZE, 3).astype("float32") * 255
    ref_probs = trained(test_input, training=False).numpy()
    ref_emb = cls_extractor(test_input, training=False).numpy()
    new_probs, new_emb = [t.numpy() for t in inference_model(test_input, training=False)]

    d_probs = float(np.max(np.abs(ref_probs - new_probs)))
    d_emb = float(np.max(np.abs(ref_emb - new_emb)))
    print(f"Max diff vs original  probs: {d_probs:.6f}   embedding: {d_emb:.6f}")
    if d_probs > 1e-4 or d_emb > 1e-4:
        raise RuntimeError("Inference graph diverges from the trained model - not converting.")
    print("Verified: inference graph exactly matches the trained model.")

    print("Converting to TFLite ...")
    inference_model.export("/tmp/dermovit_inference_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model("/tmp/dermovit_inference_saved_model")
    # Plain float32: default int8/dynamic quantization shifted softmax outputs
    # by up to ~5%, too much for a model already at ~60% accuracy. The real
    # memory win comes from the lightweight runtime, not from quantising.
    tflite_model = converter.convert()

    os.makedirs("artifacts", exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)
    print(f".keras {os.path.getsize(TRAINED_MODEL_PATH)/1e6:.1f} MB  ->  .tflite {os.path.getsize(TFLITE_OUTPUT_PATH)/1e6:.1f} MB")

    # Verify the actual .tflite file through the interpreter.
    interp = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    in_idx = interp.get_input_details()[0]["index"]
    interp.resize_tensor_input(in_idx, [1, IMG_SIZE, IMG_SIZE, 3])
    interp.allocate_tensors()
    interp.set_tensor(in_idx, test_input[:1])
    interp.invoke()
    details = interp.get_output_details()
    t_probs, t_emb = _split_outputs(details, [interp.get_tensor(d["index"]) for d in details])
    if t_probs is None or t_emb is None:
        raise RuntimeError("Expected one (1,7) and one (1,256) output in the .tflite file.")

    dp = float(np.max(np.abs(ref_probs[:1] - t_probs)))
    de = float(np.max(np.abs(ref_emb[:1] - t_emb)))
    print(f"Max diff .tflite vs original  probs: {dp:.6f}   embedding: {de:.6f}")
    if dp > 1e-2 or de > 1e-2:
        raise RuntimeError(".tflite output differs more than expected - not safe to deploy.")
    print("Verified: .tflite produces matching probabilities AND embeddings. Safe to deploy.")


if __name__ == "__main__":
    main()
