import { useState } from "react";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!file) {
      setError("Please choose an image first.");
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `Request failed (${res.status})`);
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong contacting the prediction API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #2563eb 100%)",
        padding: "30px",
        fontFamily: "Arial",
        color: "white",
      }}
    >
      <div style={{ maxWidth: "1400px", margin: "auto" }}>

        {/* HEADER */}
        <div
          style={{
            background: "rgba(255,255,255,0.1)",
            padding: "30px",
            borderRadius: "25px",
            backdropFilter: "blur(10px)",
            boxShadow: "0 10px 30px rgba(0,0,0,0.3)",
          }}
        >
          <h1 style={{ fontSize: "55px", marginBottom: "10px" }}>
            🩺 DermoViT-Lite
          </h1>

          <p style={{ fontSize: "22px", color: "#dbeafe" }}>
            Hybrid CNN + Vision Transformer Framework for Multi-Class
            Skin Cancer Classification using HAM10000 Dataset
          </p>
        </div>

        {/* STATS */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit,minmax(250px,1fr))",
            gap: "20px",
            marginTop: "30px",
          }}
        >
          {[
            ["10,015", "Dermoscopic Images"],
            ["7", "Skin Lesion Classes"],
            ["EfficientNetB0+ViT", "Model Architecture"],
            ["Real-Time", "Prediction Speed"],
          ].map((item, index) => (
            <div
              key={index}
              style={{
                background: "white",
                color: "#0f172a",
                padding: "25px",
                borderRadius: "20px",
                textAlign: "center",
                boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
              }}
            >
              <h1 style={{ fontSize: "32px", margin: 0 }}>{item[0]}</h1>
              <p style={{ fontWeight: "bold" }}>{item[1]}</p>
            </div>
          ))}
        </div>

        {/* MAIN SECTION */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "2fr 1fr",
            gap: "25px",
            marginTop: "30px",
          }}
        >
          {/* LEFT */}
          <div
            style={{
              background: "white",
              color: "#0f172a",
              padding: "30px",
              borderRadius: "25px",
              boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
            }}
          >
            <h2 style={{ fontSize: "35px" }}>
              📤 Upload Dermoscopic Image
            </h2>

            <div
              style={{
                marginTop: "20px",
                border: "3px dashed #2563eb",
                padding: "40px",
                borderRadius: "20px",
                textAlign: "center",
                background: "#eff6ff",
              }}
            >
              <h3>Upload HAM10000 Skin Lesion Image</h3>

              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{
                  marginTop: "20px",
                  fontSize: "16px",
                }}
              />

              {previewUrl && (
                <div style={{ marginTop: "20px" }}>
                  <img
                    src={previewUrl}
                    alt="Selected lesion preview"
                    style={{
                      maxWidth: "260px",
                      borderRadius: "12px",
                      boxShadow: "0 6px 16px rgba(0,0,0,0.2)",
                    }}
                  />
                </div>
              )}

              <div style={{ marginTop: "20px" }}>
                <button
                  onClick={handlePredict}
                  disabled={!file || loading}
                  style={{
                    background: !file || loading ? "#94a3b8" : "#2563eb",
                    color: "white",
                    border: "none",
                    padding: "14px 32px",
                    borderRadius: "12px",
                    fontSize: "16px",
                    fontWeight: "bold",
                    cursor: !file || loading ? "not-allowed" : "pointer",
                  }}
                >
                  {loading ? "Analyzing..." : "Run Prediction"}
                </button>
              </div>
            </div>

            {/* ERROR */}
            {error && (
              <div
                style={{
                  marginTop: "20px",
                  background: "#fee2e2",
                  color: "#991b1b",
                  padding: "18px",
                  borderRadius: "16px",
                }}
              >
                ⚠ {error}
              </div>
            )}

            {/* DEMO MODE NOTICE */}
            {result && result.demo_mode && (
              <div
                style={{
                  marginTop: "30px",
                  background: "#fef9c3",
                  padding: "20px",
                  borderRadius: "20px",
                  color: "#854d0e",
                }}
              >
                <strong>Demo mode:</strong> {result.message}
              </div>
            )}

            {/* RESULT */}
            {result && !result.demo_mode && (
              <div
                style={{
                  marginTop: "30px",
                  background: "#dcfce7",
                  padding: "25px",
                  borderRadius: "20px",
                }}
              >
                <h2 style={{ color: "#166534" }}>
                  ✅ Prediction Result
                </h2>

                <h1
                  style={{
                    color: "#15803d",
                    fontSize: "40px",
                  }}
                >
                  {result.label}
                </h1>

                <p style={{ fontSize: "20px" }}>
                  Confidence Score: <strong>{result.confidence}%</strong>
                </p>

                <p style={{ lineHeight: "1.7" }}>
                  {result.disclaimer}
                </p>

                <details style={{ marginTop: "10px" }}>
                  <summary style={{ cursor: "pointer", fontWeight: "bold" }}>
                    Full class probability breakdown
                  </summary>
                  <ul>
                    {result.all_probabilities &&
                      Object.entries(result.all_probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([cls, pct]) => (
                          <li key={cls}>
                            {cls}: {pct}%
                          </li>
                        ))}
                  </ul>
                </details>
              </div>
            )}
          </div>

          {/* RIGHT */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "25px",
            }}
          >
            {/* WORKFLOW */}
            <div
              style={{
                background: "white",
                color: "#0f172a",
                padding: "25px",
                borderRadius: "25px",
              }}
            >
              <h2>🧠 AI Workflow</h2>

              <ol style={{ lineHeight: "2" }}>
                <li>Image Upload</li>
                <li>Image Preprocessing</li>
                <li>CNN Feature Extraction</li>
                <li>Vision Transformer Analysis</li>
                <li>Hybrid Classification</li>
                <li>Prediction Output</li>
              </ol>
            </div>

            {/* TECH STACK */}
            <div
              style={{
                background: "white",
                color: "#0f172a",
                padding: "25px",
                borderRadius: "25px",
              }}
            >
              <h2>⚙ Technology Stack</h2>

              <ul style={{ lineHeight: "2" }}>
                <li>React (frontend)</li>
                <li>Flask REST API (backend)</li>
                <li>TensorFlow / Keras</li>
                <li>EfficientNetB0 (CNN backbone)</li>
                <li>Vision Transformer encoder</li>
                <li>Google Colab (training)</li>
              </ul>
            </div>
          </div>
        </div>

        {/* ARCHITECTURE */}
        <div
          style={{
            marginTop: "35px",
            background: "rgba(255,255,255,0.1)",
            padding: "30px",
            borderRadius: "25px",
            backdropFilter: "blur(8px)",
          }}
        >
          <h2 style={{ fontSize: "35px" }}>
            🏗 High Level Architecture
          </h2>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit,minmax(180px,1fr))",
              gap: "20px",
              marginTop: "25px",
            }}
          >
            {[
              "Input Image",
              "Preprocessing",
              "CNN + ViT Hybrid",
              "Softmax Classification",
              "Prediction Output",
            ].map((step, index) => (
              <div
                key={index}
                style={{
                  background: "white",
                  color: "#0f172a",
                  padding: "25px",
                  borderRadius: "20px",
                  textAlign: "center",
                  fontWeight: "bold",
                }}
              >
                {step}
              </div>
            ))}
          </div>
        </div>

        {/* FOOTER */}
        <div
          style={{
            textAlign: "center",
            marginTop: "40px",
            color: "#cbd5e1",
          }}
        >
          <p>
            DermoViT-Lite © 2026 | AIML Major Project — Phase 2
          </p>
        </div>
      </div>
    </div>
  );
}
