import { useState } from "react";

const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

export default function Auth({ onAuthenticated }) {
  const [mode, setMode] = useState("login"); // "login" | "signup"
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [message, setMessage] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage(null);
    setLoading(true);

    const endpoint = mode === "login" ? "/auth/login" : "/auth/signup";

    try {
      const res = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();

      if (!res.ok) {
        setMessage(data.message || data.error || "Something went wrong.");
        return;
      }

      if (mode === "login") {
        onAuthenticated(data.token, username);
      } else {
        setMessage("Account created — you can log in now.");
        setMode("login");
      }
    } catch (err) {
      setMessage("Could not reach the server. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        maxWidth: "380px",
        margin: "60px auto",
        background: "white",
        color: "#0f172a",
        padding: "35px",
        borderRadius: "25px",
        boxShadow: "0 10px 25px rgba(0,0,0,0.25)",
      }}
    >
      <h2 style={{ marginBottom: "20px" }}>
        {mode === "login" ? "Log In" : "Create Account"}
      </h2>

      <form onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
          style={{
            width: "100%",
            padding: "12px",
            marginBottom: "12px",
            borderRadius: "10px",
            border: "1px solid #cbd5e1",
            fontSize: "15px",
            boxSizing: "border-box",
          }}
        />
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
          minLength={mode === "signup" ? 8 : undefined}
          style={{
            width: "100%",
            padding: "12px",
            marginBottom: "12px",
            borderRadius: "10px",
            border: "1px solid #cbd5e1",
            fontSize: "15px",
            boxSizing: "border-box",
          }}
        />

        {message && (
          <div
            style={{
              background: "#fee2e2",
              color: "#991b1b",
              padding: "10px",
              borderRadius: "10px",
              marginBottom: "12px",
              fontSize: "14px",
            }}
          >
            {message}
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          style={{
            width: "100%",
            background: "#2563eb",
            color: "white",
            border: "none",
            padding: "12px",
            borderRadius: "10px",
            fontSize: "15px",
            fontWeight: "bold",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Please wait..." : mode === "login" ? "Log In" : "Sign Up"}
        </button>
      </form>

      <p style={{ marginTop: "16px", fontSize: "14px", textAlign: "center" }}>
        {mode === "login" ? "Don't have an account? " : "Already have an account? "}
        <button
          onClick={() => {
            setMode(mode === "login" ? "signup" : "login");
            setMessage(null);
          }}
          style={{
            background: "none",
            border: "none",
            color: "#2563eb",
            fontWeight: "bold",
            cursor: "pointer",
            textDecoration: "underline",
          }}
        >
          {mode === "login" ? "Sign up" : "Log in"}
        </button>
      </p>
    </div>
  );
}
