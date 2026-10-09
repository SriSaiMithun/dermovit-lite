"""Input-gate tests (validation.py). Each maps to a report test case (section 8.3)."""
from PIL import Image

import validation
from tests.conftest import make_lesion_like, make_noisy, make_screenshot


def test_lesion_like_image_passes():
    r = validation.check_image(make_lesion_like())
    assert r.ok, r.message


def test_blue_image_rejected_as_not_skin():
    r = validation.check_image(make_noisy((40, 90, 200)))
    assert (r.ok, r.code) == (False, "NOT_SKIN_COLORED")


def test_green_image_rejected_as_not_skin():
    r = validation.check_image(make_noisy((60, 160, 70)))
    assert (r.ok, r.code) == (False, "NOT_SKIN_COLORED")


def test_blank_image_rejected():
    r = validation.check_image(Image.new("RGB", (300, 300), (200, 150, 120)))
    assert (r.ok, r.code) == (False, "UNIFORM_IMAGE")


def test_screenshot_like_image_rejected():
    r = validation.check_image(make_screenshot())
    assert not r.ok and r.code in {"NOT_A_PHOTO", "TOO_BRIGHT", "NOT_SKIN_COLORED"}


def test_tiny_image_rejected():
    r = validation.check_image(Image.new("RGB", (32, 32), (200, 150, 120)))
    assert (r.ok, r.code) == (False, "TOO_SMALL")


def test_black_image_rejected():
    r = validation.check_image(make_noisy((3, 3, 3)))
    assert not r.ok and r.code in {"TOO_DARK", "UNIFORM_IMAGE"}


def test_rejection_carries_message_and_hint():
    r = validation.check_image(make_noisy((40, 90, 200)))
    assert r.message and r.hint and "skin_ratio" in r.metrics


def test_random_noise_rejected():
    import numpy as np
    rng = np.random.default_rng(0)
    r = validation.check_image(Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)))
    assert (r.ok, r.code) == (False, "NOT_A_PHOTO")


def test_realistic_textured_photo_not_flagged_as_noise():
    # the synthetic lesion has sensor-like noise and must still pass
    assert validation.check_image(make_lesion_like(seed=3)).ok
