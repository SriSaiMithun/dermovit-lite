import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import Results from "./components/Results";

beforeEach(() => {
  localStorage.clear();
  global.fetch = jest.fn(() => Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ status: "ok" }) }));
  global.URL.createObjectURL = jest.fn(() => "blob:preview");
  global.URL.revokeObjectURL = jest.fn();
});

test("shows the login screen when not signed in", () => {
  render(<App />);
  expect(screen.getByRole("heading", { name: /log in/i })).toBeInTheDocument();
  expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
  expect(screen.queryByText(/analyze image/i)).not.toBeInTheDocument();
});

test("signed-in users see the uploader, real metrics and limitations", () => {
  localStorage.setItem("dermovit_token", "t");
  localStorage.setItem("dermovit_username", "sri");
  render(<App />);
  expect(screen.getByText(/upload a lesion image/i)).toBeInTheDocument();
  expect(screen.getAllByText(/60\.3%/).length).toBeGreaterThan(0);
  expect(screen.getByText(/not a medical device/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /analyze image/i })).toBeDisabled();
});

test("rejects unsupported files on the client", async () => {
  localStorage.setItem("dermovit_token", "t");
  render(<App />);
  const bad = new File(["x"], "notes.pdf", { type: "application/pdf" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [bad] } });
  expect(await screen.findByRole("alert")).toHaveTextContent(/please upload a jpg, png or webp/i);
});

test("a 401 from the server logs the user out with a message", async () => {
  localStorage.setItem("dermovit_token", "expired");
  render(<App />);
  global.fetch = jest.fn(() => Promise.resolve({ ok: false, status: 401, json: () => Promise.resolve({}) }));
  const img = new File(["x"], "a.jpg", { type: "image/jpeg" });
  fireEvent.change(screen.getByTestId("file-input"), { target: { files: [img] } });
  fireEvent.click(screen.getByRole("button", { name: /analyze image/i }));
  await waitFor(() => expect(screen.getByRole("heading", { name: /log in/i })).toBeInTheDocument());
  expect(localStorage.getItem("dermovit_token")).toBeNull();
});

test("Results explains a rejected image without showing a prediction", () => {
  render(<Results state="rejected" result={{ message: "This looks like a screenshot.", hint: "Upload a close-up." }} />);
  expect(screen.getByText(/couldn't analyze/i)).toBeInTheDocument();
  expect(screen.getByText(/looks like a screenshot/i)).toBeInTheDocument();
  expect(screen.getByText(/no prediction was made/i)).toBeInTheDocument();
});

test("Results shows the label, risk badge and probability bars for a prediction", () => {
  const result = {
    demo_mode: false, label: "Melanocytic Nevus (mole)", risk: "low", predicted_class: "nv",
    confidence: 48.6, confidence_level: "moderate", inference_ms: 27,
    all_probabilities: { nv: 48.6, mel: 36.3, bkl: 13.7, akiec: 0.9, df: 0.4, bcc: 0.1, vasc: 0 },
    checks: { input_gate: "passed", feature_gate: "disabled" }, disclaimer: "Not a diagnosis.",
  };
  render(<Results state="ok" result={result} />);
  expect(screen.getByText("Melanocytic Nevus (mole)")).toBeInTheDocument();
  expect(screen.getByText(/lower-risk class/i)).toBeInTheDocument();
  expect(screen.getByRole("list", { name: /probability/i }).children).toHaveLength(7);
  expect(screen.getByText(/not the chance the answer is correct/i)).toBeInTheDocument();
});
