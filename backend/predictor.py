"""
Model loading, image decoding/preprocessing and inference.

Efficiency notes (all of these were per-request costs before):
- The TFLite interpreter is created and its tensors allocated ONCE at load,
  with a fixed batch size of 1, instead of resizing/re-allocating per request.
- JPEGs are decoded with PIL's draft mode, which makes the decoder shrink big
  phone photos (12 MP+) while reading, cutting time and peak memory a lot.
- EXIF orientation is applied, so phone photos aren't fed in rotated.
- A lock serialises access: a TFLite interpreter is not thread-safe.
"""

import io
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from labels import IMG_SIZE, NUM_CLASSES

Image.MAX_IMAGE_PIXELS = 50_000_000  # guard against decompression bombs


class ImageDecodeError(ValueError):
    pass


def decode_image(data: bytes) -> Image.Image:
    """Bytes -> RGB PIL image, or ImageDecodeError."""
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (1024, 1024))   # JPEG only: decode at reduced size
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageDecodeError("Image dimensions are too large.")
    except Exception:
        raise ImageDecodeError("File is not a readable image (use JPG, PNG or WebP).")


def to_model_input(img: Image.Image) -> np.ndarray:
    arr = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return arr[None, ...]  # (1, 224, 224, 3); EfficientNet preprocessing is in-graph


class Predictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._lock = threading.Lock()
        self._interp = None
        self._in_idx = None
        self._prob_idx = None
        self._emb_idx = None
        self.load_error = None
        self._load()

    @property
    def loaded(self) -> bool:
        return self._interp is not None

    @property
    def has_embedding(self) -> bool:
        return self._emb_idx is not None

    def _load(self):
        import os

        if not os.path.exists(self.model_path):
            self.load_error = f"No model file at {self.model_path}"
            return
        try:
            try:
                # Standalone interpreter: never imports TensorFlow (~130 MB vs 700 MB+).
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                try:
                    from tflite_runtime.interpreter import Interpreter
                except ImportError:
                    import tensorflow as tf
                    Interpreter = tf.lite.Interpreter

            interp = Interpreter(model_path=self.model_path)
            in_idx = interp.get_input_details()[0]["index"]
            interp.resize_tensor_input(in_idx, [1, IMG_SIZE, IMG_SIZE, 3])
            interp.allocate_tensors()   # once, not per request

            for d in interp.get_output_details():
                if d["shape"][-1] == NUM_CLASSES:
                    self._prob_idx = d["index"]
                else:
                    self._emb_idx = d["index"]   # present only in the 2-output model
            if self._prob_idx is None:
                raise RuntimeError("Model has no (1, 7) probability output.")
            self._interp, self._in_idx = interp, in_idx
        except Exception as exc:  # pragma: no cover
            self.load_error = str(exc)

    def infer(self, batch: np.ndarray):
        """Returns (probabilities[7], embedding[256] or None, milliseconds)."""
        t0 = time.perf_counter()
        with self._lock:
            self._interp.set_tensor(self._in_idx, batch)
            self._interp.invoke()
            probs = self._interp.get_tensor(self._prob_idx)[0].copy()
            emb = self._interp.get_tensor(self._emb_idx)[0].copy() if self.has_embedding else None
        return probs, emb, (time.perf_counter() - t0) * 1000.0
