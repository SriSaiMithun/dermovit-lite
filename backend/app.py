"""
DermoViT-Lite Flask inference API.

Endpoints
---------
GET  /health          -> {"status": "ok", "model_loaded": bool}
POST /predict          -> multipart/form-data with field "image"
                          returns predicted class, confidence, full
                          class probability distribution, and a short
                          clinical note (NOT a diagnosis).

Run locally:
    pip install -r requirements.txt
    python app.py
    # serves on http://localhost:5000

Uses the TFLite runtime instead of full TensorFlow for inference - this
model was originally deployed with full TensorFlow, which crashed with
out-of-memory errors on Render's free tier (512MB RAM) even after
switching to tensorflow-cpu. Converting to TFLite (see
convert_to_tflite.py) and using the lightweight tflite-runtime package
here fixed it, since it avoids loading the full TensorFlow graph/session
machinery for a model this only ever needs to run inference on.

Before running, put a converted model at
backend/artifacts/dermovit_lite.tflite (produced by convert_to_tflite.py
after training). Without it, the API starts in "demo mode" and returns a
clearly-labeled mock prediction so the frontend can still be
developed/demoed end-to-end.
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from labels import CLASS_INFO, CLASS_NAMES, IMG_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "dermovit_lite.tflite")

app = Flask(__name__)
CORS(app)  # allow the React dev server (localhost:3000) to call this API

_interpreter = None
_input_details = None
_output_details = None
_model_load_error = None


def get_interpreter():
    """Lazily loads the TFLite interpreter on first request."""
    global _interpreter, _input_details, _output_details, _model_load_error
    if _interpreter is not None or _model_load_error is not None:
        return _interpreter
    try:
        try:
            # Preferred: standalone interpreter, never imports TensorFlow
            # at all. Verified to use ~121MB peak RSS vs ~700MB when
            # falling back to full tensorflow-cpu's bundled interpreter -
            # the difference between fitting in Render's 512MB free tier
            # and getting SIGKILL'd for OOM.
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                # Last resort: works, but pulls in all of tensorflow-cpu's
                # import overhead - likely too heavy for the free tier.
                import tensorflow as tf

                Interpreter = tf.lite.Interpreter

        if os.path.exists(MODEL_PATH):
            _interpreter = Interpreter(model_path=MODEL_PATH)
            _interpreter.allocate_tensors()
            _input_details = _interpreter.get_input_details()
            _output_details = _interpreter.get_output_details()
        else:
            _model_load_error = f"No trained model found at {MODEL_PATH}"
    except Exception as exc:  # pragma: no cover
        _model_load_error = str(exc)
    return _interpreter


def preprocess_image(file_storage) -> np.ndarray:
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # preprocess_input runs inside the graph


def run_inference(batch: np.ndarray) -> np.ndarray:
    """Runs a forward pass through the TFLite interpreter, returning the
    (7,) softmax probability array for a single image.
    """
    interpreter = get_interpreter()
    interpreter.resize_tensor_input(_input_details[0]["index"], batch.shape)
    interpreter.allocate_tensors()
    interpreter.set_tensor(_input_details[0]["index"], batch)
    interpreter.invoke()
    return interpreter.get_tensor(_output_details[0]["index"])[0]


@app.route("/health", methods=["GET"])
def health():
    interpreter = get_interpreter()
    return jsonify({"status": "ok", "model_loaded": interpreter is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    interpreter = get_interpreter()

    try:
        batch = preprocess_image(file)
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    if interpreter is None:
        # Demo mode — no trained/converted model yet. Makes it obvious in
        # the response that this is not a real prediction, so it's never
        # confused with a trained-model result during evaluation/demo.
        return jsonify(
            {
                "demo_mode": True,
                "message": _model_load_error
                or "Model not loaded — run train.py then convert_to_tflite.py, and place the .tflite file in backend/artifacts/",
                "predicted_class": None,
                "confidence": None,
            }
        ), 200

    probs = run_inference(batch)
    top_idx = int(np.argmax(probs))
    top_class = CLASS_NAMES[top_idx]

    response = {
        "demo_mode": False,
        "predicted_class": top_class,
        "label": CLASS_INFO[top_class]["label"],
        "risk": CLASS_INFO[top_class]["risk"],
        "confidence": round(float(probs[top_idx]) * 100, 2),
        "all_probabilities": {
            CLASS_NAMES[i]: round(float(p) * 100, 2) for i, p in enumerate(probs)
        },
        "disclaimer": (
            "This is an AI-assisted screening prediction, not a medical "
            "diagnosis. Please consult a dermatologist for confirmation."
        ),
    }
    return jsonify(response), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
