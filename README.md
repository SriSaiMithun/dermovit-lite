# DermoViT-Lite

Hybrid CNN + Vision Transformer model for multi-class skin lesion classification on the HAM10000 dataset, with a Flask REST API backend and React frontend.

> **Academic prototype - not a medical device.** It must not be used for diagnosis. Always consult a dermatologist.

| | |
|---|---|
| Frontend (Vercel) | https://dermovit-lite-blond.vercel.app |
| Backend (Render)  | https://dermovit-lite-backend.onrender.com/health |

The backend runs on a free tier that sleeps when idle, so the first request after a quiet period can take up to a minute.

## Features
- Upload a lesion image, get a probability distribution over 7 HAM10000 classes with a risk badge.
- JWT authentication (signup/login, salted scrypt password hashing); `/predict` requires a token.
- **Rejects images that aren't skin-lesion photos** instead of returning a confident wrong answer (see below).
- Lightweight inference: TFLite + `ai-edge-litert`, ~145 MB RAM and ~20 ms per image on a desktop CPU (slower on the shared free tier), no TensorFlow at serve time.
- Responsive UI with drag-and-drop, client-side downscaling of large photos, and server wake-up status.

## Architecture
```
 React (Vercel)  --HTTPS-->  Flask API (Render)
   login, upload               /auth/*   JWT + SQLite users
   result UI                   /predict  decode -> input gate -> TFLite model -> feature gate -> JSON
                               /health
 Training (Colab, GPU):  train.py -> convert_to_tflite.py -> fit_ood_detector.py -> artifacts/
```
Model: EfficientNetB0 backbone -> 1x1 projection -> 49 patch tokens + CLS token -> 4-layer Transformer encoder (4 heads) -> dense head -> softmax(7).

## Results (held-out test split, 1,503 images)
| Metric | Value |
|---|---|
| Accuracy | 60.3% |
| Macro F1 | 0.444 |

Per-class results are in `backend/artifacts/metrics.json` and shown in the app. The dataset is heavily imbalanced (~67% benign nevi), so rare classes are weak (melanoma precision 23.9%, dermatofibroma F1 0.15). These are honest numbers for a short training run, not clinical performance.

## Handling non-skin images (out-of-distribution input)
A 7-class classifier always answers with one of its 7 classes. Left alone, uploading a cat photo returns "melanoma".

**Softmax confidence cannot fix this.** Measured on the trained model: a colour wheel scored 99.8% confidence, a face photo 96.2% and a screenshot 96.5%, while a genuine dermoscopy image scored only 48.6%. A "reject low confidence" rule would reject real lesions and accept junk. So the API uses two checks that don't depend on confidence:

1. **Input gate** (`validation.py`) - cheap, explainable image checks: minimum size, blank/uniform, too dark/bright, screenshot/graphic detection (large perfectly-flat regions), random noise, and fraction of skin/lesion-coloured pixels. On the images I tested (38 screenshots plus grass, gravel, brick, text, colour wheel, coins, moon, noise and solid colours) it rejected every one, while passing a real dermoscopy image with a wide margin. It deliberately cannot reject skin-coloured photos: a face, a brown cat, a coffee cup and a retina image all passed it.
2. **Feature-distance gate** (`ood.py`) - Mahalanobis distance between the image's 256-d CLS embedding and class-conditional Gaussians fitted on real HAM10000 images; the threshold is the 99th percentile of held-out validation scores. This is what catches cats, food and faces. It needs a one-time calibration (below) and is disabled until `artifacts/ood_stats.npz` exists (`/health` reports `feature_gate_enabled`).

Limits: both are heuristics. The calibration script reports the real false-rejection rate on held-out HAM10000 test images and how non-skin photos are handled; quote those numbers, not the claims here.

## API
All `/predict` responses include `status`.

| Endpoint | Description |
|---|---|
| `GET /health` | `{status, model_loaded, feature_gate_enabled}` |
| `POST /auth/signup` | `{username, password(>=8)}` -> 201 |
| `POST /auth/login` | `{username, password}` -> `{token}` (24 h) |
| `POST /predict` | multipart `image` (JPG/PNG/WebP, <= 8 MB), header `Authorization: Bearer <token>` |

`/predict` -> `200 {"status":"ok","predicted_class","label","risk","confidence","confidence_level","all_probabilities","checks","inference_ms","disclaimer"}`
or `422 {"status":"rejected","code","message","hint"}` with codes `TOO_SMALL, UNIFORM_IMAGE, TOO_DARK, TOO_BRIGHT, NOT_A_PHOTO, NOT_SKIN_COLORED, OUT_OF_DISTRIBUTION`. Other errors: 400 (bad file), 401 (token), 413 (too large).

## Run locally
```bash
# backend
cd backend
pip install -r requirements.txt
set JWT_SECRET_KEY=any-long-random-string      # PowerShell: $env:JWT_SECRET_KEY="..."
python app.py                                  # http://localhost:5000

# frontend (new terminal, repo root)
copy .env.example .env                         # REACT_APP_API_URL=http://localhost:5000
npm install
npm start
```

## Train, convert, calibrate (Google Colab, GPU)
```bash
pip install -q kagglehub scikit-learn pandas matplotlib ai-edge-litert scikit-image
python train.py              # -> artifacts/dermovit_lite.keras, metrics.json
python convert_to_tflite.py  # -> artifacts/dermovit_lite.tflite (probabilities + embedding), verified against the .keras
python fit_ood_detector.py   # -> artifacts/ood_stats.npz, ood_report.json
```
Download the three artifacts into `backend/artifacts/` and redeploy. Download files immediately - Colab storage is wiped when the session ends.

## Tests
```bash
cd backend && pip install -r requirements-dev.txt && python -m pytest -q   # 25 tests: gates, API, auth, limits
cd .. && CI=true npm test -- --watchAll=false                              # 6 UI tests
```

## Configuration
| Variable | Where | Purpose |
|---|---|---|
| `JWT_SECRET_KEY` | Render | **Required in production** - signs login tokens |
| `CORS_ORIGINS` | Render | Comma-separated allowed origins, e.g. the Vercel URL (default `*`) |
| `PYTHON_VERSION` | Render | e.g. `3.11.9` |
| `REACT_APP_API_URL` | Vercel / `.env` | Backend base URL (baked in at build time - redeploy after changing) |
| `GATE_*` | Render (optional) | Override input-gate thresholds (see `validation.py`) |

## Known limitations
- Trained on dermoscopic images only; ordinary phone photos of skin will be unreliable.
- SQLite user store lives on Render's ephemeral disk: accounts are lost on each redeploy. A managed database is the production fix.
- Free-tier cold starts (~1 min).
- No rate limiting on the API.

## Dataset & citation
HAM10000 - Tschandl P., Rosendahl C., Kittler H., *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions*, Sci. Data 5, 180161 (2018). Licensed CC BY-NC 4.0 (non-commercial).
