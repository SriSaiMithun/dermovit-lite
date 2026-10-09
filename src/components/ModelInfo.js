import metrics from "../metrics.json";
import { CLASSES } from "../classes";

const pct = (x) => `${(x * 100).toFixed(1)}%`;

export default function ModelInfo() {
  const rows = Object.entries(metrics.per_class).sort((a, b) => b[1]["f1-score"] - a[1]["f1-score"]);
  return (
    <section className="card" aria-labelledby="perf-title">
      <h2 id="perf-title" className="card-title">Model performance (held-out test set)</h2>
      <div className="stat-row">
        <div><strong>{pct(metrics.test_accuracy)}</strong><span>Accuracy</span></div>
        <div><strong>{metrics.macro_f1.toFixed(2)}</strong><span>Macro F1</span></div>
        <div><strong>1,503</strong><span>Test images</span></div>
      </div>
      <table className="perf-table">
        <thead><tr><th>Class</th><th>F1</th><th>Precision</th><th>Recall</th><th>Images</th></tr></thead>
        <tbody>
          {rows.map(([code, m]) => (
            <tr key={code}>
              <td>{CLASSES[code].name}</td>
              <td>
                <span className="bar-track mini"><span className={`bar-fill risk-${CLASSES[code].risk}`} style={{ width: `${m["f1-score"] * 100}%` }} /></span>
                {m["f1-score"].toFixed(2)}
              </td>
              <td>{pct(m.precision)}</td>
              <td>{pct(m.recall)}</td>
              <td>{m.support}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">
        HAM10000 is heavily imbalanced (about two-thirds are benign moles), so rare classes are much harder -
        melanoma precision is only {pct(metrics.per_class.mel.precision)}. That is why this tool is a learning
        prototype and must never be used for real diagnosis.
      </p>
    </section>
  );
}
