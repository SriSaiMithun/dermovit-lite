"""
One-time calibration of the out-of-distribution gates on real HAM10000 data.

Run in Colab/Kaggle. It mirrors train.py's dataset loading and uses the exact same
train/val/test split (same seed), so the validation and test images were never seen
by the model - essential for an honest threshold:

    pip install -q ai-edge-litert scikit-image
    python convert_to_tflite.py      # once: makes the 2-output (probs + embedding) model
    python fit_ood_detector.py       # ~10 min on a Colab CPU

What it does
  1. Embeds a stratified subset of TRAIN images with the production TFLite
     model and production preprocessing (same code path as the live API).
  2. Fits class-conditional Gaussians with a shared, shrunk covariance.
  3. Sets the rejection threshold at the Nth percentile (default 99) of scores
     on held-out VALIDATION images -> ~1% of genuine images are rejected.
  4. Measures the real false-rejection rate on held-out TEST images, for BOTH
     gates (input gate in validation.py and the feature gate).
  5. Shows what the gates do to natural non-skin photos (cat, coffee, face...).

Outputs (copy both into backend/artifacts/ and redeploy):
    artifacts/ood_stats.npz      statistics used by ood.py at inference time
    artifacts/ood_report.json    numbers to quote in the report (section 8.4)
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import validation
from labels import CLASS_NAMES
from predictor import Predictor, decode_image, to_model_input
SEED = 42  # must equal train.py's SEED so the val/test split matches the one used for training
OUT_DIR = "artifacts"
SHRINKAGE = 0.05


def find_dataset_root() -> str:
    """Kaggle Notebook input if mounted, else download via kagglehub (needs kaggle.json)."""
    if os.path.isdir("/kaggle/input"):
        for name in os.listdir("/kaggle/input"):
            cand = os.path.join("/kaggle/input", name)
            if os.path.exists(os.path.join(cand, "HAM10000_metadata.csv")):
                return cand
    import kagglehub

    return kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")


def load_metadata(dataset_root: str) -> pd.DataFrame:
    """Mirrors train.py's load_metadata EXACTLY (same row order after dropna), because the
    stratified split below must reproduce the one train.py used."""
    df = pd.read_csv(os.path.join(dataset_root, "HAM10000_metadata.csv"))
    lookup = {}
    for d in (os.path.join(dataset_root, "HAM10000_images_part_1"),
              os.path.join(dataset_root, "HAM10000_images_part_2")):
        if os.path.isdir(d):
            for fname in os.listdir(d):
                lookup[fname.split(".")[0]] = os.path.join(d, fname)
    df["image_path"] = df["image_id"].map(lookup)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df["label"] = df["dx"]
    return df[["image_path", "label"]]


def embed_paths(predictor, paths, desc):
    """Returns (embeddings[N,D], input_gate_rejections[N] bool)."""
    embs, rejected = [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        with open(p, "rb") as f:
            img = decode_image(f.read())
        rejected.append(not validation.check_image(img).ok)
        _, emb, _ = predictor.infer(to_model_input(img))
        embs.append(emb)
        if (i + 1) % 250 == 0 or i + 1 == len(paths):
            print(f"  {desc}: {i + 1}/{len(paths)}  ({time.time() - t0:.0f}s)")
    return np.stack(embs), np.array(rejected)


def stratified_subset(df, n_total, min_per_class=60):
    parts = []
    for _, g in df.groupby("label"):
        k = min(len(g), max(min_per_class, round(n_total * len(g) / len(df))))
        parts.append(g.sample(n=k, random_state=SEED))
    return pd.concat(parts)


def mahalanobis_min(x, means, precision):
    diffs = x[:, None, :] - means[None, :, :]                  # (N, K, D)
    d2 = np.einsum("nkd,de,nke->nk", diffs, precision, diffs)  # (N, K)
    return d2.min(axis=1)


def natural_photo_check(predictor, means, precision, threshold):
    """Score some non-skin photos bundled with scikit-image (if installed)."""
    try:
        import skimage.data as sd
        from PIL import Image
    except ImportError:
        return None
    rows = {}
    for name in ["astronaut", "coffee", "chelsea", "rocket", "camera", "grass", "gravel",
                 "brick", "retina", "immunohistochemistry", "colorwheel", "text", "page", "coins"]:
        try:
            arr = np.asarray(getattr(sd, name)())
        except Exception:
            continue
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, -1)
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = (255 * arr / arr.max()).astype(np.uint8)
        img = Image.fromarray(arr)
        g1 = validation.check_image(img)
        _, emb, _ = predictor.infer(to_model_input(img))
        score = float(mahalanobis_min(emb[None], means, precision)[0])
        rows[name] = {
            "input_gate": "rejected:" + g1.code if not g1.ok else "passed",
            "feature_distance": round(score, 1),
            "feature_gate": "rejected" if score > threshold else "passed",
            "rejected_by_either": (not g1.ok) or score > threshold,
        }
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--max-val", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=1000)
    ap.add_argument("--percentile", type=float, default=99.0)
    ap.add_argument("--model", default=os.path.join(OUT_DIR, "dermovit_lite.tflite"))
    args = ap.parse_args()

    predictor = Predictor(args.model)
    if not predictor.loaded:
        raise SystemExit(f"Could not load model: {predictor.load_error}")
    if not predictor.has_embedding:
        raise SystemExit("This .tflite has no embedding output - run convert_to_tflite.py first.")

    from sklearn.model_selection import train_test_split

    df = load_metadata(args.data_root or find_dataset_root())
    # Identical split to train.py (same seed/stratification) -> val/test unseen by the model.
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED)

    train_sub = stratified_subset(train_df, args.max_train)
    val_sub = val_df.sample(n=min(args.max_val, len(val_df)), random_state=SEED)
    test_sub = test_df.sample(n=min(args.max_test, len(test_df)), random_state=SEED)
    print(f"train={len(train_sub)} val={len(val_sub)} test={len(test_sub)} images")

    tr_x, tr_rej = embed_paths(predictor, train_sub["image_path"].tolist(), "train")
    tr_y = train_sub["label"].map({c: i for i, c in enumerate(CLASS_NAMES)}).values

    means = np.stack([tr_x[tr_y == k].mean(axis=0) for k in range(len(CLASS_NAMES))])
    resid = tr_x - means[tr_y]
    cov = np.cov(resid, rowvar=False)
    d = cov.shape[0]
    cov = (1 - SHRINKAGE) * cov + SHRINKAGE * (np.trace(cov) / d) * np.eye(d)
    precision = np.linalg.inv(cov)

    va_x, va_rej = embed_paths(predictor, val_sub["image_path"].tolist(), "val")
    va_scores = mahalanobis_min(va_x, means, precision)
    threshold = float(np.percentile(va_scores, args.percentile))

    te_x, te_rej = embed_paths(predictor, test_sub["image_path"].tolist(), "test")
    te_scores = mahalanobis_min(te_x, means, precision)

    feat_fr = float(np.mean(te_scores > threshold))
    in_fr = float(np.mean(te_rej))
    either_fr = float(np.mean((te_scores > threshold) | te_rej))
    photos = natural_photo_check(predictor, means, precision, threshold)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        os.path.join(OUT_DIR, "ood_stats.npz"),
        means=means, precision=precision, threshold=threshold,
        percentile=args.percentile, shrinkage=SHRINKAGE,
        n_train=len(train_sub), n_val=len(val_sub),
    )
    report = {
        "threshold": threshold,
        "percentile": args.percentile,
        "val_score_quantiles": {str(q): float(np.percentile(va_scores, q)) for q in (50, 90, 95, 99)},
        "held_out_test_false_rejection": {
            "n": len(test_sub),
            "input_gate": in_fr,
            "feature_gate": feat_fr,
            "either_gate": either_fr,
        },
        "non_skin_photos": photos,
    }
    with open(os.path.join(OUT_DIR, "ood_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Calibration results ===")
    print(f"Threshold (squared Mahalanobis, {args.percentile:g}th pct of val): {threshold:.1f}")
    print(f"False rejection of GENUINE held-out test images ({len(test_sub)}):")
    print(f"   input gate   : {in_fr * 100:.2f}%   (should be ~0% - if not, loosen GATE_* env thresholds)")
    print(f"   feature gate : {feat_fr * 100:.2f}%   (expected ~{100 - args.percentile:g}%)")
    print(f"   either gate  : {either_fr * 100:.2f}%")
    if photos:
        caught = sum(v["rejected_by_either"] for v in photos.values())
        print(f"\nNon-skin photos rejected: {caught}/{len(photos)}")
        for k, v in photos.items():
            print(f"   {k:22s} input={v['input_gate']:26s} feature_dist={v['feature_distance']:9.1f} -> {v['feature_gate']}")
    print(f"\nSaved {OUT_DIR}/ood_stats.npz and {OUT_DIR}/ood_report.json - copy into backend/artifacts/ and redeploy.")


if __name__ == "__main__":
    main()
