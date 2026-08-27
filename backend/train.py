"""
Training script for DermoViT-Lite on the HAM10000 dataset.

DESIGNED TO BE RUN ON GOOGLE COLAB (with a GPU runtime), not on a laptop.
Runtime > Change runtime type > GPU (T4 is enough).

Steps this script performs:
  1. Download HAM10000 from Kaggle via kagglehub.
  2. Build a dataframe of (image_path, label) from HAM10000_metadata.csv.
  3. Split into train/val/test (stratified) — 70/15/15.
  4. Build a tf.data pipeline.
  5. Compute class weights (HAM10000 is heavily imbalanced — ~67% is `nv`).
  6. Build the hybrid CNN+ViT model from model.py.
  7. Train in two stages: (a) frozen backbone, (b) fine-tune with backbone
     unfrozen at a low LR.
  8. Evaluate on the held-out test set: accuracy, precision/recall/F1 per
     class, confusion matrix, and save the plots + a metrics.json — these
     numbers are what go into report section 8.4 (AI Model Validation).
  9. Save the trained model as dermovit_lite.keras for the Flask backend.

Usage (in a Colab cell):
    !pip install -q kagglehub
    !python train.py
    # or just run the cells if you paste this into a notebook
"""

import json
import os

import kagglehub
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import CLASS_NAMES, IMG_SIZE, build_model

SEED = 42
BATCH_SIZE = 32
EPOCHS_STAGE1 = 15   # frozen backbone
EPOCHS_STAGE2 = 10   # fine-tuning
OUTPUT_DIR = "artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Computes macro-F1 on a held-out tf.data set after every epoch and
    logs it as `val_macro_f1`, so ModelCheckpoint/EarlyStopping can select
    the model that's actually best on the minority classes — not just the
    model that's best at predicting the majority `nv` class, which is what
    `val_accuracy` alone rewards on an imbalanced dataset like HAM10000.
    """

    def __init__(self, val_ds, num_classes):
        super().__init__()
        self.val_ds = val_ds
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        y_true, y_pred = [], []
        for imgs, labels in self.val_ds:
            probs = self.model.predict(imgs, verbose=0)
            y_pred.extend(np.argmax(probs, axis=1).tolist())
            y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
        macro_f1 = f1_score(
            y_true, y_pred, average="macro", labels=list(range(self.num_classes)), zero_division=0
        )
        logs["val_macro_f1"] = macro_f1
        print(f" — val_macro_f1: {macro_f1:.4f}")


def load_metadata(dataset_root: str) -> pd.DataFrame:
    """Builds a dataframe with columns [image_path, label] from HAM10000."""
    meta_path = os.path.join(dataset_root, "HAM10000_metadata.csv")
    df = pd.read_csv(meta_path)

    # Images are split across two folders in the Kaggle release.
    img_dirs = [
        os.path.join(dataset_root, "HAM10000_images_part_1"),
        os.path.join(dataset_root, "HAM10000_images_part_2"),
    ]
    path_lookup = {}
    for d in img_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                path_lookup[fname.split(".")[0]] = os.path.join(d, fname)

    df["image_path"] = df["image_id"].map(path_lookup)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    df["label"] = df["dx"]  # dx column already uses akiec/bcc/bkl/df/mel/nv/vasc
    return df[["image_path", "label"]]


def make_dataset(df: pd.DataFrame, class_to_idx: dict, training: bool) -> tf.data.Dataset:
    paths = df["image_path"].values
    labels = df["label"].map(class_to_idx).values

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        return img, tf.one_hot(label, depth=len(class_to_idx))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    print("Downloading HAM10000 from Kaggle (requires kaggle.json credentials)...")
    dataset_root = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
    print("Dataset at:", dataset_root)

    df = load_metadata(dataset_root)
    print("Total labeled images:", len(df))
    print(df["label"].value_counts())

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=SEED
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=SEED
    )
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = make_dataset(train_df, class_to_idx, training=True)
    val_ds = make_dataset(val_df, class_to_idx, training=False)
    test_ds = make_dataset(test_df, class_to_idx, training=False)

    # Class imbalance: HAM10000 is ~67% `nv`. Weight the loss so rare,
    # clinically important classes (mel, akiec, bcc) aren't ignored.
    class_weights_arr = compute_class_weight(
        "balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_df["label"].map(class_to_idx).values,
    )
    class_weights = dict(enumerate(class_weights_arr))
    print("Class weights:", class_weights)

    model = build_model(freeze_backbone=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    stage1_ckpt = os.path.join(OUTPUT_DIR, "stage1_best.keras")
    stage2_ckpt = os.path.join(OUTPUT_DIR, "stage2_best.keras")

    def make_callbacks(ckpt_path):
        # Fresh callback instances per stage — otherwise a re-used
        # ModelCheckpoint/EarlyStopping silently compares stage-2 val
        # scores against stage-1's best, and (worse) restore_best_weights
        # can hand stage 2 a "best" that's still worse than stage 1 ever
        # reached, which is exactly what happened on the first run.
        #
        # Monitoring val_macro_f1 (not val_accuracy) so checkpointing and
        # early stopping select the model that's actually best across all
        # 7 classes, not just best at the majority `nv` class.
        return [
            MacroF1Callback(val_ds, num_classes=len(CLASS_NAMES)),
            tf.keras.callbacks.ModelCheckpoint(
                ckpt_path, monitor="val_macro_f1", mode="max", save_best_only=True
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max", patience=5, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        ]

    print("\n=== Stage 1: training classifier head + ViT on top of frozen CNN ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=make_callbacks(stage1_ckpt),
    )
    stage1_best_val_f1 = max(history1.history["val_macro_f1"])

    print("\n=== Stage 2: fine-tuning with backbone unfrozen (low LR) ===")
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    # Keep BatchNorm layers frozen even though the backbone is unfrozen.
    # Updating BN running statistics with our small batch size (32) on a
    # dataset this size destroys the pretrained features almost
    # immediately — this is what caused stage 2 to collapse from ~60%
    # to ~44% val accuracy on the first run.
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=make_callbacks(stage2_ckpt),
    )
    stage2_best_val_f1 = max(history2.history["val_macro_f1"])

    # ---- Pick whichever stage actually produced the better model ----
    # Fine-tuning is *supposed* to help, but if it doesn't (still possible
    # even with BN frozen, depending on your data), don't ship the worse
    # model just because it was trained more recently. Compared on
    # macro-F1 now, matching what we actually optimized for above.
    if stage2_best_val_f1 >= stage1_best_val_f1:
        best_ckpt = stage2_ckpt
        print(f"\nStage 2 improved on stage 1 ({stage2_best_val_f1:.4f} >= {stage1_best_val_f1:.4f}) — using stage 2 model.")
    else:
        best_ckpt = stage1_ckpt
        print(f"\nStage 2 did NOT improve on stage 1 ({stage2_best_val_f1:.4f} < {stage1_best_val_f1:.4f}) — keeping stage 1 model.")

    model = tf.keras.models.load_model(best_ckpt, compile=False)

    # ---- Save final model for the Flask backend ----
    final_path = os.path.join(OUTPUT_DIR, "dermovit_lite.keras")
    model.save(final_path)
    print("Saved model to", final_path)

    # ---- Evaluate on held-out test set ----
    y_true, y_pred, y_prob = [], [], []
    for imgs, labels in test_ds:
        probs = model.predict(imgs, verbose=0)
        y_prob.extend(probs.tolist())
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred).tolist()
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {
        "test_accuracy": report["accuracy"],
        "macro_f1": macro_f1,
        "per_class": {c: report[c] for c in CLASS_NAMES},
        "confusion_matrix": cm,
        "class_order": CLASS_NAMES,
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Test set results (use these numbers in report section 8.4) ===")
    print(json.dumps({"test_accuracy": metrics["test_accuracy"], "macro_f1": macro_f1}, indent=2))

    # Save training curves for the report (loss/accuracy vs epoch).
    try:
        import matplotlib.pyplot as plt

        for key in ["accuracy", "loss"]:
            plt.figure()
            plt.plot(history1.history[key] + history2.history[key], label="train")
            plt.plot(history1.history[f"val_{key}"] + history2.history[f"val_{key}"], label="val")
            plt.axvline(EPOCHS_STAGE1, color="gray", linestyle="--", label="fine-tune start")
            plt.xlabel("epoch")
            plt.ylabel(key)
            plt.legend()
            plt.title(f"DermoViT-Lite {key} curve")
            plt.savefig(os.path.join(OUTPUT_DIR, f"{key}_curve.png"))
    except ImportError:
        pass

    print(f"\nAll artifacts (model, metrics.json, curves) saved in ./{OUTPUT_DIR}/")
    print("Download dermovit_lite.keras and put it in backend/artifacts/ next to app.py")


if __name__ == "__main__":
    main()
