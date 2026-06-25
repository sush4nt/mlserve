import { useState } from "react";
import { infer } from "../services/api";
import ResultDisplay from "./ResultDisplay.jsx";

// Build the full ordered feature vector from the editable values, filling
// non-editable features with their training-median defaults.
export function assembleVector(model, values) {
  return model.fields.map((f) =>
    f.editable ? Number(values[f.name] ?? f.default) : Number(f.default)
  );
}

export default function PredictionForm({ model }) {
  const editable = model.fields.filter((f) => f.editable);
  const [values, setValues] = useState(
    Object.fromEntries(editable.map((f) => [f.name, f.default]))
  );
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const vec = assembleVector(model, values);
      setResult(await infer(model.id, vec, model.datatype));
    } catch (e) {
      setError(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="rounded-xl bg-slate-800 p-6">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {editable.map((f) => (
          <label key={f.name} className="text-sm">
            <span className="text-slate-400">{f.label}</span>
            <input
              type="number"
              value={values[f.name]}
              min={f.min ?? undefined}
              max={f.max ?? undefined}
              step="any"
              onChange={(e) => setValues({ ...values, [f.name]: e.target.value })}
              className="mt-1 w-full rounded bg-slate-900 px-2 py-1 font-mono text-slate-100 outline-none ring-1 ring-slate-700 focus:ring-blue-500"
            />
          </label>
        ))}
      </div>

      <button
        onClick={run}
        disabled={busy}
        className="mt-4 rounded-lg bg-blue-600 px-4 py-2 font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {busy ? "Predicting…" : "Predict"}
      </button>
      {result && (
        <span className="ml-3 font-mono text-sm text-slate-400">
          {result.latencyMs.toFixed(1)} ms
        </span>
      )}

      {error && <p className="mt-3 text-sm text-rose-400">Error: {error}</p>}
      <ResultDisplay model={model} result={result} />
    </div>
  );
}
