# DermoViT-Lite backend

Flask API + model code. See the repository root `README.md` for the full guide (architecture, API, training,
calibration, deployment, limitations).

| File | Purpose |
|---|---|
| `app.py` | API routes and the predict pipeline |
| `predictor.py` | Image decoding + TFLite inference (model loaded once) |
| `validation.py` | Input gate: rejects non-skin / unusable images |
| `ood.py` | Feature-distance gate (Mahalanobis) |
| `auth.py` | Signup/login, password hashing, JWT |
| `model.py`, `labels.py` | Architecture (training) and shared constants |
| `train.py`, `convert_to_tflite.py`, `fit_ood_detector.py` | Colab pipeline: train -> convert -> calibrate |
| `tests/` | pytest suite |
