"""
Input validation gate for DermoViT-Lite (layer 1 of out-of-distribution handling).

WHY THIS EXISTS
---------------
The classifier always outputs one of 7 skin-lesion classes, even for images
that are not skin at all. Worse, measurements showed its softmax confidence
is NOT a usable signal for this: it was 96-99% confident on a colour wheel,
a face photo and a screenshot, but only ~49% on a real dermoscopic image.
So "reject low-confidence predictions" would reject real lesions and accept
junk. Instead we check the *image itself* with cheap, explainable tests.

WHAT IT DOES AND DOES NOT CATCH
-------------------------------
Catches (high precision): screenshots / UI / text / graphics, random noise, blank or
uniform images, near-black or blown-out images, images with almost no
skin/lesion-coloured pixels (sky, grass, grey, blue, ...).

Does NOT catch: photos that happen to be skin-coloured (a brown cat, a face,
a coffee cup). That needs the feature-distance gate in ood.py, which is fitted
on the real dataset (see fit_ood_detector.py).

The thresholds are deliberately conservative: it is much worse to reject a
valid lesion image than to let an odd image through to the second gate.
Run calibrate via fit_ood_detector.py on real HAM10000 images and confirm the
rejection rate is ~0% before trusting them. All values can be overridden with
environment variables.
"""

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


MIN_SIDE_PX = int(_env_float("GATE_MIN_SIDE_PX", 64))
MIN_LUMA_STD = _env_float("GATE_MIN_LUMA_STD", 8.0)        # below: blank / uniform
MIN_MEAN_LUMA = _env_float("GATE_MIN_MEAN_LUMA", 25.0)      # below: nearly black
MAX_MEAN_LUMA = _env_float("GATE_MAX_MEAN_LUMA", 248.0)     # above: blown out
MAX_FLAT_RATIO = _env_float("GATE_MAX_FLAT_RATIO", 0.50)    # above: graphic / screenshot
MIN_SKIN_RATIO = _env_float("GATE_MIN_SKIN_RATIO", 0.10)    # below: not skin-coloured
MAX_ROUGHNESS = _env_float("GATE_MAX_ROUGHNESS", 0.90)     # above: random noise / static


@dataclass
class GateResult:
    ok: bool
    code: str = "OK"
    message: str = ""
    hint: str = ""
    metrics: dict = field(default_factory=dict)


HINT = (
    "Upload a clear, well-lit, close-up photo (ideally dermoscopic) of a single "
    "skin lesion."
)


def compute_metrics(img: Image.Image) -> dict:
    """Cheap image statistics, computed on a <=256px thumbnail."""
    small = img.convert("RGB")
    small.thumbnail((256, 256))

    hsv = np.asarray(small.convert("HSV"), dtype=np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # PIL HSV is 0-255. Skin / lesion tones sit in the red-orange-yellow band
    # (roughly 0-50 degrees, wrapping to ~335-360) with at least a little colour.
    skin_like = ((h <= 36) | (h >= 238)) & (s >= 10) & (s <= 240) & (v >= 40)

    gray = np.asarray(small.convert("L"), dtype=np.float32)
    gx = np.abs(np.diff(gray, axis=1))[:-1, :]
    gy = np.abs(np.diff(gray, axis=0))[:, :-1]
    # Photos have sensor noise / texture; screenshots and graphics have large
    # perfectly flat regions (zero gradient in both directions).
    flat_ratio = float(np.mean((gx < 1) & (gy < 1)))
    # Mean neighbour difference relative to overall contrast. Real photos are spatially smooth
    # (measured 0.05-0.63); white noise / static is ~1.15.
    roughness = float(np.abs(np.diff(gray, axis=1)).mean() / (gray.std() + 1e-6))

    return {
        "width": img.size[0],
        "height": img.size[1],
        "skin_ratio": round(float(skin_like.mean()), 4),
        "luma_std": round(float(gray.std()), 2),
        "mean_luma": round(float(gray.mean()), 2),
        "flat_ratio": round(flat_ratio, 4),
        "roughness": round(roughness, 3),
    }


def check_image(img: Image.Image) -> GateResult:
    """Returns GateResult(ok=False, ...) for images that clearly are not a
    usable skin-lesion photo. Order matters: cheapest / most certain first.
    """
    m = compute_metrics(img)

    if min(m["width"], m["height"]) < MIN_SIDE_PX:
        return GateResult(
            False, "TOO_SMALL",
            f"Image is too small ({m['width']}x{m['height']} px).",
            f"Use an image at least {MIN_SIDE_PX}px on each side.", m,
        )
    if m["luma_std"] < MIN_LUMA_STD:
        return GateResult(
            False, "UNIFORM_IMAGE",
            "This image is blank or a single flat colour.", HINT, m,
        )
    if m["mean_luma"] < MIN_MEAN_LUMA:
        return GateResult(
            False, "TOO_DARK", "This image is almost completely black.",
            "Retake the photo with better lighting.", m,
        )
    if m["mean_luma"] > MAX_MEAN_LUMA:
        return GateResult(
            False, "TOO_BRIGHT", "This image is almost completely white.",
            "Retake the photo without flash glare or overexposure.", m,
        )
    if m["flat_ratio"] > MAX_FLAT_RATIO:
        return GateResult(
            False, "NOT_A_PHOTO",
            "This looks like a screenshot, drawing or graphic, not a photo of skin.",
            HINT, m,
        )
    if m["roughness"] > MAX_ROUGHNESS:
        return GateResult(
            False, "NOT_A_PHOTO", "This image looks like random noise or static, not a photo.", HINT, m,
        )
    if m["skin_ratio"] < MIN_SKIN_RATIO:
        return GateResult(
            False, "NOT_SKIN_COLORED",
            "This image doesn't look like skin - it contains almost no skin-toned pixels.",
            HINT, m,
        )
    return GateResult(True, metrics=m)
