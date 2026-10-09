// All network access lives here so components stay simple and testable.
export const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_INPUT_MB = 20;
const MAX_SIDE_PX = 1024;

/** Client-side check before we upload anything. Returns an error string or null. */
export function validateFile(file) {
  if (!file) return "Please choose an image first.";
  if (!ALLOWED_TYPES.includes(file.type)) return "Please upload a JPG, PNG or WebP image.";
  if (file.size > MAX_INPUT_MB * 1024 * 1024) return `Image is too large (max ${MAX_INPUT_MB} MB).`;
  return null;
}

/**
 * Shrinks big phone photos before upload (max 1024px on the long side, JPEG).
 * The model only sees 224x224, so this loses nothing useful, but it makes uploads
 * much faster and keeps memory low on the free-tier server. Falls back to the
 * original file if the browser can't do it.
 */
export async function prepareImage(file) {
  try {
    if (typeof createImageBitmap !== "function") return file;
    const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
    const scale = Math.min(1, MAX_SIDE_PX / Math.max(bitmap.width, bitmap.height));
    if (scale === 1 && file.size < 1.5 * 1024 * 1024) return file;
    const canvas = document.createElement("canvas");
    canvas.width = Math.round(bitmap.width * scale);
    canvas.height = Math.round(bitmap.height * scale);
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.92));
    return blob || file;
  } catch {
    return file;
  }
}

/** Wakes/checks the backend. Resolves true if it answered. */
export async function pingHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

/** mode: "login" | "signup". Returns { ok, token?, message? } */
export async function authRequest(mode, username, password) {
  try {
    const res = await fetch(`${API_BASE_URL}/auth/${mode}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) return { ok: false, message: data.message || data.error || "Something went wrong." };
    return { ok: true, token: data.token, message: data.message };
  } catch {
    return { ok: false, message: "Could not reach the server. It may be waking up - please try again in a moment." };
  }
}

/**
 * Returns one of:
 *   { kind: "ok", data }            prediction
 *   { kind: "rejected", data }      not a usable skin-lesion image (HTTP 422)
 *   { kind: "unauthorized" }        token missing/expired
 *   { kind: "error", message }
 */
export async function predict(file, token) {
  try {
    const body = new FormData();
    body.append("image", await prepareImage(file), "lesion.jpg");
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { Authorization: `Bearer ${token}` },
      body,
    });
    const data = await res.json().catch(() => ({}));
    if (res.status === 401) return { kind: "unauthorized" };
    if (res.status === 422) return { kind: "rejected", data };
    if (!res.ok) return { kind: "error", message: data.error || `Request failed (${res.status}).` };
    return { kind: "ok", data };
  } catch {
    return { kind: "error", message: "Could not reach the server. Please check your connection and try again." };
  }
}
