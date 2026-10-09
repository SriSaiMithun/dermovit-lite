import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";
import Auth from "./Auth";
import ModelInfo from "./components/ModelInfo";
import Results from "./components/Results";
import Uploader from "./components/Uploader";
import { pingHealth, predict, validateFile } from "./api";
import metrics from "./metrics.json";

const SERVER_LABEL = {
  checking: "Connecting…",
  waking: "Server waking up (free tier)…",
  ready: "Server ready",
  offline: "Server unreachable",
};

function useServerStatus() {
  const [status, setStatus] = useState("checking");
  useEffect(() => {
    let alive = true;
    const slow = setTimeout(() => alive && setStatus((s) => (s === "checking" ? "waking" : s)), 2500);
    pingHealth().then((ok) => alive && setStatus(ok ? "ready" : "offline"));
    return () => { alive = false; clearTimeout(slow); };
  }, []);
  return status;
}

export default function App() {
  const [token, setToken] = useState(() => localStorage.getItem("dermovit_token"));
  const [username, setUsername] = useState(() => localStorage.getItem("dermovit_username"));
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileError, setFileError] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | loading | ok | rejected | error
  const [result, setResult] = useState(null);
  const [errorMsg, setErrorMsg] = useState(null);
  const [notice, setNotice] = useState(null);
  const server = useServerStatus();
  const urlRef = useRef(null);

  const clearImage = useCallback(() => {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = null;
    setFile(null); setPreviewUrl(null); setFileError(null);
    setPhase("idle"); setResult(null); setErrorMsg(null);
  }, []);

  useEffect(() => () => urlRef.current && URL.revokeObjectURL(urlRef.current), []);

  const handleAuthenticated = (newToken, name) => {
    localStorage.setItem("dermovit_token", newToken);
    localStorage.setItem("dermovit_username", name);
    setToken(newToken); setUsername(name); setNotice(null);
  };

  const handleLogout = useCallback((message = null) => {
    localStorage.removeItem("dermovit_token");
    localStorage.removeItem("dermovit_username");
    clearImage();
    setToken(null); setUsername(null); setNotice(message);
  }, [clearImage]);

  const handleFile = (f) => {
    const problem = validateFile(f);
    if (problem) { setFileError(problem); return; }
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = URL.createObjectURL(f);
    setFile(f); setPreviewUrl(urlRef.current); setFileError(null);
    setPhase("idle"); setResult(null);
  };

  const handleAnalyze = async () => {
    const problem = validateFile(file);
    if (problem) { setFileError(problem); return; }
    setPhase("loading"); setResult(null); setErrorMsg(null);
    const out = await predict(file, token);
    if (out.kind === "unauthorized") return handleLogout("Your session expired - please log in again.");
    if (out.kind === "ok") { setResult(out.data); setPhase("ok"); }
    else if (out.kind === "rejected") { setResult(out.data); setPhase("rejected"); }
    else { setErrorMsg(out.message); setPhase("error"); }
  };

  if (!token) {
    return (
      <div className="app app-auth">
        <Auth onAuthenticated={handleAuthenticated} notice={notice} serverStatus={server} />
      </div>
    );
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="brand-mark" aria-hidden="true" />DermoViT-Lite</div>
        <div className="topbar-right">
          <span className={`status status-${server}`}>{SERVER_LABEL[server]}</span>
          <span className="muted small">Signed in as <strong>{username}</strong></span>
          <button className="btn btn-ghost btn-sm" onClick={() => handleLogout()}>Log out</button>
        </div>
      </header>

      <main className="container">
        <section className="hero">
          <h1>Skin lesion screening with a hybrid CNN + Vision Transformer</h1>
          <p>
            An academic prototype trained on the HAM10000 dermoscopy dataset. It looks at one lesion image and
            estimates which of 7 lesion types it most resembles.
          </p>
          <div className="stats">
            <div><strong>10,015</strong><span>training images</span></div>
            <div><strong>7</strong><span>lesion classes</span></div>
            <div><strong>{(metrics.test_accuracy * 100).toFixed(1)}%</strong><span>test accuracy</span></div>
            <div><strong>{metrics.macro_f1.toFixed(2)}</strong><span>macro F1</span></div>
          </div>
        </section>

        <div className="grid-2">
          <Uploader
            previewUrl={previewUrl}
            fileName={file?.name}
            loading={phase === "loading"}
            error={fileError}
            onFile={handleFile}
            onAnalyze={handleAnalyze}
            onClear={clearImage}
          />
          <Results state={phase} result={result} message={errorMsg} />
        </div>

        <ModelInfo />

        <section className="card" aria-labelledby="how-title">
          <h2 id="how-title" className="card-title">How it works</h2>
          <ol className="steps">
            <li><strong>Image check</strong><span>Rejects screenshots, blank, too dark/bright and non-skin images.</span></li>
            <li><strong>CNN features</strong><span>EfficientNetB0 extracts local texture and colour patterns.</span></li>
            <li><strong>Vision Transformer</strong><span>A 4-layer encoder relates regions across the whole lesion.</span></li>
            <li><strong>Resemblance check</strong><span>Compares the image's features with real training data.</span></li>
            <li><strong>Classification</strong><span>Softmax over 7 classes, shown as probabilities.</span></li>
          </ol>
        </section>

        <section className="card card-warn" aria-labelledby="limits-title">
          <h2 id="limits-title" className="card-title">Limitations &amp; responsible use</h2>
          <ul className="plain-list">
            <li>This is <strong>not a medical device</strong> and not a diagnosis. See a dermatologist for any concerning skin change.</li>
            <li>It was trained only on dermoscopic images from one dataset; phone photos of skin will be less reliable.</li>
            <li>Rare classes (e.g. melanoma, dermatofibroma) are weak - see the per-class table above.</li>
            <li>The image checks are heuristics and can occasionally reject a valid image or accept an odd one.</li>
          </ul>
        </section>
      </main>

      <footer className="footer">DermoViT-Lite © 2026 · AI &amp; ML Major Project, Phase 2</footer>
    </div>
  );
}
