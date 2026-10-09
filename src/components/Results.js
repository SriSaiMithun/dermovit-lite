import { CLASSES } from "../classes";

const LEVEL_TEXT = {
  high: "High model confidence",
  moderate: "Moderate model confidence",
  low: "Low model confidence",
};

function Bars({ probabilities, topClass }) {
  return (
    <ul className="bars" aria-label="Probability for each class">
      {Object.entries(probabilities).map(([code, pct]) => (
        <li key={code} className={code === topClass ? "is-top" : ""}>
          <span className="bar-label">{CLASSES[code]?.name || code}</span>
          <span className="bar-track"><span className={`bar-fill risk-${CLASSES[code]?.risk}`} style={{ width: `${Math.max(pct, 0.5)}%` }} /></span>
          <span className="bar-pct">{pct.toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

export default function Results({ state, result, message }) {
  return (
    <section className="card result-card" aria-labelledby="result-title" aria-live="polite">
      <h2 id="result-title" className="card-title">2 · Result</h2>

      {state === "idle" && (
        <div className="empty">
          <p>Your result will appear here.</p>
          <p className="muted small">Every image is first checked to make sure it actually looks like a skin-lesion photo before the model sees it.</p>
        </div>
      )}

      {state === "loading" && (
        <div className="loading" role="status">
          <div className="spinner" aria-hidden="true" />
          <p>Analyzing image…</p>
          <p className="muted small">The first request after a quiet period can take up to a minute while the free-tier server wakes up.</p>
        </div>
      )}

      {state === "rejected" && result && (
        <div className="alert alert-warn" role="alert">
          <h3>We couldn't analyze this image</h3>
          <p>{result.message}</p>
          {result.hint && <p className="muted">{result.hint}</p>}
          <p className="small muted">No prediction was made - the model is only meant for skin-lesion photos, and guessing on anything else would be misleading.</p>
        </div>
      )}

      {state === "error" && <div className="alert alert-error" role="alert">{message}</div>}

      {state === "ok" && result && result.demo_mode && (
        <div className="alert alert-warn">Demo mode: {result.message}</div>
      )}

      {state === "ok" && result && !result.demo_mode && (
        <div className="prediction">
          <div className="pred-head">
            <div>
              <p className="muted small">Most likely class</p>
              <h3 className="pred-label">{result.label}</h3>
            </div>
            <span className={`badge badge-${result.risk}`}>{result.risk === "high" ? "Higher-risk class" : "Lower-risk class"}</span>
          </div>

          <div className="confidence">
            <div className="confidence-row">
              <span>{LEVEL_TEXT[result.confidence_level]}</span>
              <strong>{result.confidence.toFixed(1)}%</strong>
            </div>
            <div className="bar-track big"><span className={`bar-fill level-${result.confidence_level}`} style={{ width: `${result.confidence}%` }} /></div>
            <p className="muted small">Model confidence is not the chance the answer is correct - this model scores about 60% accuracy overall (see below).</p>
          </div>

          <Bars probabilities={result.all_probabilities} topClass={result.predicted_class} />

          <div className="checks">
            <span className="chip ok">✓ Image check passed</span>
            {result.checks?.feature_gate === "passed" && <span className="chip ok">✓ Resembles training data</span>}
            <span className="chip">{result.inference_ms} ms</span>
          </div>

          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}
    </section>
  );
}
