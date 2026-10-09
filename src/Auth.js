import { useState } from "react";
import { authRequest } from "./api";

export default function Auth({ onAuthenticated, notice, serverStatus }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState(notice || null);
  const [success, setSuccess] = useState(false);
  const [loading, setLoading] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    setMessage(null); setSuccess(false); setLoading(true);
    const out = await authRequest(mode, username.trim(), password);
    setLoading(false);
    if (!out.ok) { setMessage(out.message); return; }
    if (mode === "login") onAuthenticated(out.token, username.trim());
    else { setMode("login"); setPassword(""); setSuccess(true); setMessage("Account created - you can log in now."); }
  };

  const switchMode = () => { setMode(mode === "login" ? "signup" : "login"); setMessage(null); setSuccess(false); };

  return (
    <div className="auth-card card">
      <div className="brand brand-lg"><span className="brand-mark" aria-hidden="true" />DermoViT-Lite</div>
      <p className="muted">Skin lesion screening prototype. Sign in to continue.</p>
      <h1 className="auth-title">{mode === "login" ? "Log in" : "Create account"}</h1>

      <form onSubmit={submit}>
        <label htmlFor="username">Username</label>
        <input id="username" autoComplete="username" value={username} onChange={(e) => setUsername(e.target.value)} required />
        <label htmlFor="password">Password {mode === "signup" && <span className="muted small">(min. 8 characters)</span>}</label>
        <input
          id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)}
          autoComplete={mode === "login" ? "current-password" : "new-password"}
          required minLength={mode === "signup" ? 8 : undefined}
        />
        {message && <div className={`alert ${success ? "alert-ok" : "alert-error"}`} role="alert">{message}</div>}
        <button className="btn btn-primary btn-block" type="submit" disabled={loading}>
          {loading ? "Please wait…" : mode === "login" ? "Log in" : "Sign up"}
        </button>
      </form>

      {serverStatus === "waking" && (
        <p className="muted small center">The free-tier server is waking up - the first login can take up to a minute.</p>
      )}
      <p className="center small">
        {mode === "login" ? "No account yet? " : "Already registered? "}
        <button type="button" className="link-btn" onClick={switchMode}>{mode === "login" ? "Sign up" : "Log in"}</button>
      </p>
    </div>
  );
}
