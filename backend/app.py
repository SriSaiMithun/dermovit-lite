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

Before running, put a trained model at backend/artifacts/dermovit_lite.keras
(produced by train.py on Colab). Without it, the API starts in "demo mode"
and returns a clearly-labeled mock prediction so the frontend can still be
developed/demoed end-to-end.
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from model import CLASS_INFO, CLASS_NAMES, IMG_SIZE

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "dermovit_lite.keras")

app = Flask(__name__)
CORS(app)  # allow the React dev server (localhost:3000) to call this API

_model = None
_model_load_error = None


def get_model():
    """Lazily loads the trained Keras model on first request."""
    global _model, _model_load_error
    if _model is not None or _model_load_error is not None:
        return _model
    try:
        import tensorflow as tf

        if os.path.exists(MODEL_PATH):
            _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        else:
            _model_load_error = f"No trained model found at {MODEL_PATH}"
    except Exception as exc:  # pragma: no cover
        _model_load_error = str(exc)
    return _model


def preprocess_image(file_storage) -> np.ndarray:
    img = Image.open(io.BytesIO(file_storage.read())).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # model.py's preprocess_input runs inside the graph


@app.route("/health", methods=["GET"])
def health():
    model = get_model()
    return jsonify({"status": "ok", "model_loaded": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No 'image' file part in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    model = get_model()

    try:
        batch = preprocess_image(file)
    except Exception as exc:
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    if model is None:
        # Demo mode — no trained weights yet. Makes it obvious in the
        # response that this is not a real prediction, so it's never
        # confused with a trained-model result during evaluation/demo.
        return jsonify(
            {
                "demo_mode": True,
                "message": _model_load_error
                or "Model not loaded — run train.py and place the .keras file in backend/artifacts/",
                "predicted_class": None,
                "confidence": None,
            }
        ), 200

    probs = model.predict(batch, verbose=0)[0]
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
