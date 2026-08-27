# DermoViT-Lite Backend

Flask REST API that serves the hybrid CNN + Vision Transformer skin
lesion classifier to the React frontend.

## Files
- `model.py` — hybrid CNN (EfficientNetB0) + ViT encoder architecture.
- `train.py` — training script, **meant to run on Google Colab** (GPU),
  not locally. Downloads HAM10000 via `kagglehub`, trains in two stages
  (frozen backbone → fine-tune), saves `artifacts/dermovit_lite.keras`
  and `artifacts/metrics.json`.
- `app.py` — Flask API (`/health`, `/predict`).
- `requirements.txt` — backend Python dependencies.

## 1. Train the model (Google Colab)
1. Open a new Colab notebook, set Runtime → GPU (T4 is enough).
2. Upload `model.py` and `train.py` to the Colab session (or `git clone`
   your repo there).
3. Get a Kaggle API token (kaggle.com → Account → Create New API Token,
   downloads `kaggle.json`) and upload it to Colab, then run:
   ```python
   !mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   !pip install -q kagglehub tensorflow scikit-learn pandas matplotlib
   !python train.py
   ```
4. This takes roughly 30–60 minutes on a T4 GPU for ~25 total epochs.
5. Download `artifacts/dermovit_lite.keras` and `artifacts/metrics.json`
   from Colab and place them in `backend/artifacts/` in this repo.
   `metrics.json` has the accuracy/F1/confusion-matrix numbers to quote
   in the report's "AI Model Validation" section.

## 2. Run the API locally
```bash
cd backend
python -m venv venv && source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python app.py
```
Runs on `http://localhost:5000`. Without a trained model in
`artifacts/`, `/predict` responds in a clearly labeled `demo_mode` so
the frontend still works end-to-end while the model trains.

## 3. Point the frontend at it
In `dermovit-lite/.env` (copy from `.env.example`):
```
REACT_APP_API_URL=http://localhost:5000
```
Then `npm start` in the frontend as usual.

## API reference
### `GET /health`
```json
{ "status": "ok", "model_loaded": true }
```

### `POST /predict`  (multipart/form-data, field name `image`)
```json
{
  "demo_mode": false,
  "predicted_class": "mel",
  "label": "Melanoma",
  "risk": "high",
  "confidence": 91.42,
  "all_probabilities": { "akiec": 1.2, "bcc": 2.1, "...": "..." },
  "disclaimer": "This is an AI-assisted screening prediction, not a medical diagnosis..."
}
```
