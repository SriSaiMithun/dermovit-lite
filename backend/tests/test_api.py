"""API tests: auth, upload handling, gates and response shape."""
import io
import os

import pytest

import app as app_module
from tests.conftest import make_lesion_like, make_noisy, to_bytes

needs_model = pytest.mark.skipif(
    not os.path.exists(app_module.MODEL_PATH), reason="trained .tflite not present"
)


def post(client, token, data, name="x.jpg", mime="image/jpeg"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return client.post("/predict", data={"image": (io.BytesIO(data), name, mime)},
                       content_type="multipart/form-data", headers=headers)


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json()["status"] == "ok"


def test_predict_requires_token(client):
    assert post(client, None, to_bytes(make_lesion_like())).status_code == 401


def test_predict_rejects_garbage_token(client):
    assert post(client, "garbage.token.here", to_bytes(make_lesion_like())).status_code == 401


def test_signup_validation_and_duplicates(client):
    assert client.post("/auth/signup", json={"username": "a", "password": "short"}).status_code == 400
    assert client.post("/auth/signup", json={"username": "dup", "password": "password123"}).status_code == 201
    assert client.post("/auth/signup", json={"username": "dup", "password": "password123"}).status_code == 400


def test_login_wrong_password(client):
    client.post("/auth/signup", json={"username": "u1", "password": "password123"})
    assert client.post("/auth/login", json={"username": "u1", "password": "nope"}).status_code == 401


def test_non_image_file_is_400(client, token):
    r = post(client, token, b"this is not an image", "x.jpg", "image/jpeg")
    assert r.status_code == 400


def test_unsupported_mimetype_is_400(client, token):
    r = post(client, token, b"%PDF-1.4", "x.pdf", "application/pdf")
    assert r.status_code == 400


def test_missing_file_is_400(client, token):
    r = client.post("/predict", data={}, content_type="multipart/form-data",
                    headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 400


def test_oversized_upload_is_413(client, token):
    r = post(client, token, os.urandom(9 * 1024 * 1024), "big.png", "image/png")
    assert r.status_code == 413


@needs_model
def test_non_skin_image_is_rejected_with_422(client, token):
    r = post(client, token, to_bytes(make_noisy((40, 90, 200))))
    body = r.get_json()
    assert r.status_code == 422
    assert body["status"] == "rejected" and body["code"] == "NOT_SKIN_COLORED"
    assert body["message"] and body["hint"]


@needs_model
def test_valid_image_returns_full_prediction(client, token):
    r = post(client, token, to_bytes(make_lesion_like()))
    body = r.get_json()
    assert r.status_code == 200 and body["status"] == "ok"
    assert body["predicted_class"] in {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}
    assert abs(sum(body["all_probabilities"].values()) - 100) < 0.5
    assert body["confidence_level"] in {"low", "moderate", "high"}
    assert body["checks"]["input_gate"] == "passed" and body["inference_ms"] > 0


@needs_model
def test_png_and_webp_are_accepted(client, token):
    img = make_lesion_like()
    assert post(client, token, to_bytes(img, "PNG"), "a.png", "image/png").status_code == 200
    assert post(client, token, to_bytes(img, "WEBP"), "a.webp", "image/webp").status_code == 200
