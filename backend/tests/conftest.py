import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_lesion_like(size=300, seed=0) -> Image.Image:
    """Synthetic dermoscopy-like image: pale pink skin + dark brown lesion + sensor noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size * 0.28)
    blob = np.clip(1.6 - r, 0, 1)[..., None]
    skin = np.array([236, 205, 195], dtype=np.float32)
    lesion = np.array([110, 62, 38], dtype=np.float32)
    arr = skin * (1 - blob) + lesion * blob + rng.normal(0, 4, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_noisy(color, size=300, seed=1, variation=45) -> Image.Image:
    """Solid-ish colour with a smooth lighting gradient + noise, like a real photo of a
    sky / lawn / wall (a perfectly flat image would be caught as UNIFORM_IMAGE instead)."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(-variation, variation, size, dtype=np.float32)[None, :, None]
    arr = np.array(color, dtype=np.float32) + ramp + rng.normal(0, 6, (size, size, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def make_screenshot(size=400) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    px = img.load()
    for y in range(40, 360, 22):          # fake "text lines"
        for x in range(30, 300):
            if (x // 6) % 3:
                px[x, y] = px[x, y + 1] = (30, 30, 30)
    return img


def to_bytes(img: Image.Image, fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    import auth
    monkeypatch.setattr(auth, "DB_PATH", str(tmp_path / "users.db"))
    auth.init_db()
    import app as app_module
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


@pytest.fixture
def token(client):
    client.post("/auth/signup", json={"username": "tester", "password": "password123"})
    return client.post("/auth/login", json={"username": "tester", "password": "password123"}).get_json()["token"]
