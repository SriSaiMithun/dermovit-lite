"""
Converts the trained dermovit_lite.keras model into a lightweight
dermovit_lite.tflite file for low-memory deployment (e.g. Render's free
tier, which crashed running full TensorFlow inference at ~512MB RAM).

Run this once, locally, after training:
    python convert_to_tflite.py

It rebuilds the inference graph by reusing the ORIGINAL trained model's
layer objects directly (same weights, no transfer/copy needed), skipping
only the 4 training-only data augmentation layers (RandomFlip/Rotation/
Zoom/Contrast). Those are identity at inference time anyway, but leaving
them in forces the TFLite converter to pull in heavy "Flex" TF ops for
no benefit. The script verifies outputs match the original model before
writing artifacts/dermovit_lite.tflite.
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

import model as model_module  # registers custom layers (BroadcastClsToken etc.) - must import before load_model

TRAINED_MODEL_PATH = os.path.join("artifacts", "dermovit_lite.keras")
TFLITE_OUTPUT_PATH = os.path.join("artifacts", "dermovit_lite.tflite")
IMG_SIZE = 224


def build_inference_model_from_trained(trained: tf.keras.Model) -> tf.keras.Model:
    """Rewires the trained model's own layers into a fresh graph that
    skips the 4 augmentation layers, reusing every other layer object
    (and its already-trained weights) as-is - no weight copying needed.
    """
    skip_names = {"lesion_image", "random_flip", "random_rotation", "random_zoom", "random_contrast"}
    kept_layers = [l for l in trained.layers if l.name not in skip_names]

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="lesion_image_infer")
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    # Everything from the backbone onward is a straight chain EXCEPT the
    # cls-token branch, which needs [cls_token, tokens] as a pair input.
    idx = 0
    # 1) backbone
    x = kept_layers[idx](x); idx += 1          # efficientnetb0
    # 2) conv projection + reshape to token sequence
    x = kept_layers[idx](x); idx += 1          # conv2d
    tokens = kept_layers[idx](x); idx += 1     # reshape

    # 3) cls token branch
    broadcast_layer = kept_layers[idx]; idx += 1   # broadcast_cls_token
    concat_layer = kept_layers[idx]; idx += 1      # concatenate
    # The Embedding layer producing the raw cls token only ever took a
    # constant (tf.zeros) as input, so Keras froze its output as a literal
    # constant array at save time rather than keeping it as a live layer -
    # it doesn't even appear in trained.layers or as a traceable `.input`.
    # Pull the frozen value directly out of the layer's stored call node.
    node = broadcast_layer._inbound_nodes[0]
    cls_token_value = node.arguments.args[0][0]  # the constant cls_token tensor
    cls_token_const = tf.constant(cls_token_value)

    cls_token = broadcast_layer([cls_token_const, tokens])
    tokens = concat_layer([cls_token, tokens])

    # 4) positional embedding + transformer blocks + head (straight chain)
    while idx < len(kept_layers):
        tokens = kept_layers[idx](tokens)
        idx += 1

    return models.Model(inputs, tokens, name="DermoViT-Lite-Inference")


def main():
    print(f"Loading trained model from {TRAINED_MODEL_PATH} ...")
    trained = tf.keras.models.load_model(TRAINED_MODEL_PATH, compile=False)

    print("Rebuilding inference-only graph (reusing trained layers, skipping augmentation) ...")
    inference_model = build_inference_model_from_trained(trained)

    # Verify: same input should give (near) identical output on both models.
    test_input = (np.random.rand(3, IMG_SIZE, IMG_SIZE, 3).astype("float32")) * 255
    orig_out = trained(test_input, training=False).numpy()
    new_out = inference_model(test_input, training=False).numpy()
    max_diff = np.max(np.abs(orig_out - new_out))
    print(f"Max output difference between original and inference model: {max_diff:.6f}")
    if max_diff > 1e-4:
        raise RuntimeError(
            "Inference model outputs diverge from the trained model - graph "
            "rewiring likely has a bug. Not proceeding with TFLite conversion."
        )
    print("Verified: inference model exactly matches trained model. Proceeding with conversion.")

    print("Converting to TFLite ...")
    inference_model.export("/tmp/dermovit_inference_saved_model")
    converter = tf.lite.TFLiteConverter.from_saved_model(
        "/tmp/dermovit_inference_saved_model"
    )
    # Deliberately skip aggressive (e.g. DEFAULT/int8) quantization here -
    # it shifted softmax outputs by up to ~5%, more than acceptable for a
    # model already at 60% accuracy. The real memory win for deployment
    # comes from using the lightweight TFLite runtime instead of full
    # TensorFlow, not from quantizing weights, so plain float32 conversion
    # is the safer choice.
    tflite_model = converter.convert()

    os.makedirs("artifacts", exist_ok=True)
    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    orig_size = os.path.getsize(TRAINED_MODEL_PATH) / 1e6
    tflite_size = os.path.getsize(TFLITE_OUTPUT_PATH) / 1e6
    print(f"Original .keras size: {orig_size:.1f} MB")
    print(f"Converted .tflite size: {tflite_size:.1f} MB")

    # Final sanity check: run the actual .tflite file through the
    # lightweight interpreter and confirm it also matches.
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    single_input = test_input[:1]
    interpreter.resize_tensor_input(input_details[0]["index"], single_input.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]["index"], single_input)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(output_details[0]["index"])

    tflite_diff = np.max(np.abs(orig_out[:1] - tflite_out))
    print(f"Max output difference between original and .tflite file: {tflite_diff:.6f}")
    if tflite_diff > 1e-2:
        raise RuntimeError(".tflite output differs more than expected - not safe to deploy.")
    print("Verified: .tflite file produces matching predictions. Safe to deploy.")


if __name__ == "__main__":
    main()

