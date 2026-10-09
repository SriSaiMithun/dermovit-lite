import { useRef, useState } from "react";

export default function Uploader({ previewUrl, fileName, loading, error, onFile, onAnalyze, onClear }) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.[0]) onFile(e.dataTransfer.files[0]);
  };

  return (
    <section className="card" aria-labelledby="upload-title">
      <h2 id="upload-title" className="card-title">1 · Upload a lesion image</h2>

      <div
        className={`dropzone ${dragging ? "is-dragging" : ""} ${previewUrl ? "has-image" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => !previewUrl && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if ((e.key === "Enter" || e.key === " ") && !previewUrl) inputRef.current?.click(); }}
        aria-label="Choose or drop an image"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          hidden
          data-testid="file-input"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
        {previewUrl ? (
          <figure className="preview">
            <img src={previewUrl} alt="Selected lesion preview" />
            <figcaption>{fileName}</figcaption>
          </figure>
        ) : (
          <div className="drop-hint">
            <div className="drop-icon" aria-hidden="true">⬆</div>
            <p><strong>Drag &amp; drop</strong> an image here, or <span className="link">browse</span></p>
            <p className="muted small">JPG, PNG or WebP · best with a clear, close-up dermoscopic photo of one lesion</p>
          </div>
        )}
      </div>

      {error && <div className="alert alert-error" role="alert">{error}</div>}

      <div className="actions">
        <button className="btn btn-primary" onClick={onAnalyze} disabled={!previewUrl || loading}>
          {loading ? "Analyzing…" : "Analyze image"}
        </button>
        {previewUrl && !loading && (
          <button className="btn btn-ghost" onClick={() => { onClear(); if (inputRef.current) inputRef.current.value = ""; }}>
            Choose another
          </button>
        )}
      </div>
    </section>
  );
}
