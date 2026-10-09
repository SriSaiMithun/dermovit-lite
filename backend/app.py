"""
DermoViT-Lite Flask API.

Endpoints
---------
GET  /health         public. Model + gate status.
POST /auth/signup    {username, password}
POST /auth/login     {username, password} -> {token}
POST /predict        multipart "image", header "Authorization: Bearer <token>"

/predict pipeline
-----------------
  decode  ->  input gate (validation.py)  ->  model  ->  feature gate (ood.py)  ->  respond

Responses from /predict always carry a "status":
  "ok"        200  prediction returned
  "rejected"  422  image is not a usable skin-lesion photo (code + message + hint)
  (errors)    400/401/413  malformed request, bad token, file too large

The model is a classifier over 7 lesion classes, so it would otherwise answer
"melanoma" for a photo of a cat. The two gates exist to prevent that. See
validation.py and ood.py for why softmax confidence is not used for this.
"""

import os
from functools import wraps

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

import auth
import validation
from labels import CLASS_INFO, CLASS_NAMES
from ood import FeatureGate
from predictor import ImageDecodeError, Predictor, decode_image, to_model_input
from labels import CLASS_INFO, CLASS_NAMES, IMG_SIZE

HERE = os.path.dirname(__file__)
MODEL_PATH = os.path.join(HERE, "artifacts", "dermovit_lite.tflite")
OOD_STATS_PATH = os.path.join(HERE, "artifacts", "ood_stats.npz")
MAX_UPLOAD_MB = 8
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
# Set CORS_ORIGINS on Render to your Vercel URL to lock the API to your frontend.
CORS(app, origins=[o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",")])

if auth.JWT_SECRET == "dev-only-insecure-secret-change-me":
    app.logger.warning("JWT_SECRET_KEY is not set - using an insecure development secret!")

_predictor = None
_feature_gate = None
_loaded = False


def get_predictor() -> Predictor:
    """Loads the model (and optional feature gate) lazily on first use."""
    global _predictor, _feature_gate, _loaded
    if not _loaded:
        _predictor = Predictor(MODEL_PATH)
        _feature_gate = FeatureGate.load_if_available(OOD_STATS_PATH)
        _loaded = True
    return _predictor


def confidence_level(pct: float) -> str:
    return "high" if pct >= 70 else "moderate" if pct >= 45 else "low"


def rejected(code: str, message: str, hint: str = "", metrics: dict | None = None):
    body = {"status": "rejected", "code": code, "message": message, "hint": hint}
    if metrics is not None:
        body["metrics"] = metrics
    return jsonify(body), 422


def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return jsonify({"error": "Missing or malformed Authorization header."}), 401
        valid, result = auth.verify_token(header.removeprefix("Bearer ").strip())
        if not valid:
            return jsonify({"error": result}), 401
        request.username = result
        return f(*args, **kwargs)

    return wrapper


@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": f"File too large (max {MAX_UPLOAD_MB} MB)."}), 413



@app.route("/auth/signup", methods=["POST"])
def signup():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    success, message = auth.create_user(username, password)
    return jsonify({"success": success, "message": message}), (201 if success else 400)


@app.route("/auth/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    success, result = auth.verify_user(username, password)
    if not success:
        return jsonify({"success": False, "message": result}), 401
    return jsonify({"success": True, "token": result}), 200


@app.route("/health", methods=["GET"])
def health():
    p = get_predictor()
    return jsonify({
        "status": "ok",
        "model_loaded": p.loaded,
        "feature_gate_enabled": _feature_gate is not None,
    })



@app.route("/predict", methods=["POST"])
@require_auth
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No 'image' file in request."}), 400
    if file.mimetype not in ALLOWED_TYPES:
        return jsonify({"error": "Unsupported file type. Use JPG, PNG or WebP."}), 400

    predictor = get_predictor()
    if not predictor.loaded:
        return jsonify({
            "status": "ok", "demo_mode": True,
            "message": predictor.load_error or "Model not loaded.",
            "predicted_class": None, "confidence": None,
        }), 200

    try:
        img = decode_image(file.read())
    except ImageDecodeError as exc:
        return jsonify({"error": str(exc)}), 400

    # Gate 1: is this plausibly a skin photo at all? (cheap, no model needed)
    gate = validation.check_image(img)
    if not gate.ok:
        return rejected(gate.code, gate.message, gate.hint, gate.metrics)

    probs, embedding, ms = predictor.infer(to_model_input(img))

    # Gate 2: does it look like the training data in the model's feature space?
    feature_gate = "disabled"
    if _feature_gate is not None and embedding is not None:
        in_dist, score = _feature_gate.is_in_distribution(embedding)
        if not in_dist:
            return rejected(
                "OUT_OF_DISTRIBUTION",
                "This image doesn't resemble the dermoscopic skin-lesion images the model was trained on.",
                "Upload a clear, close-up photo of a single skin lesion.",
                {"feature_distance": round(score, 1), "threshold": round(_feature_gate.threshold, 1)},
            )
        feature_gate = "passed"

    order = np.argsort(probs)[::-1]
    top = CLASS_NAMES[int(order[0])]
    pct = round(float(probs[order[0]]) * 100, 2)
    return jsonify({
        "status": "ok",
        "demo_mode": False,
        "predicted_class": top,
        "label": CLASS_INFO[top]["label"],
        "risk": CLASS_INFO[top]["risk"],
        "confidence": pct,
        "confidence_level": confidence_level(pct),
        "all_probabilities": {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in order},
        "checks": {"input_gate": "passed", "feature_gate": feature_gate},
        "inference_ms": round(ms, 1),
        "disclaimer": (
            "This is an AI-assisted screening prediction, not a medical diagnosis. "
            "Please consult a dermatologist for confirmation."
        ),
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
