import { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { infer } from "../services/api";
import { assembleVector } from "./PredictionForm.jsx";

// Sends the same input to both Avazu endpoints simultaneously and shows the
// client-measured latency side by side. The latency gap is small for a single
// request — the k6 load test is where ONNX pulls clearly ahead.
export default function ComparisonPanel({ pyModel, onnxModel }) {
  const [results, setResults] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async () => {
    setBusy(true);
    const vecPy = assembleVector(pyModel, {});
    const vecOnnx = assembleVector(onnxModel, {});
    const [py, onnx] = await Promise.all([
      infer(pyModel.id, vecPy, pyModel.datatype),
      infer(onnxModel.id, vecOnnx, onnxModel.datatype),
    ]);
    setResults({ py, onnx });
    setBusy(false);
  };

  const data = results
    ? [
        { name: "Python", latency: results.py.latencyMs, prob: results.py.prediction, color: "#3b82f6" },
        { name: "ONNX/C++", latency: results.onnx.latencyMs, prob: results.onnx.prediction, color: "#22c55e" },
      ]
    : [];

  return (
    <div className="rounded-xl bg-slate-800 p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-white">Runtime Comparison</h2>
        <button
          onClick={run}
          disabled={busy}
          className="rounded-lg bg-indigo-600 px-4 py-2 font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
        >
          {busy ? "Running…" : "Run Both"}
        </button>
      </div>

      {results && (
        <div className="mt-4 grid grid-cols-2 gap-4">
          {data.map((d) => (
            <div key={d.name} className="rounded-lg bg-slate-900 p-4">
              <p className="text-sm text-slate-400">{d.name}</p>
              <p className="font-mono text-2xl text-white">{d.latency.toFixed(1)} ms</p>
              <p className="text-sm text-slate-400">P(click) = {d.prob.toFixed(4)}</p>
            </div>
          ))}
          <div className="col-span-2 h-44">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data}>
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis unit="ms" stroke="#94a3b8" />
                <Tooltip
                  contentStyle={{ background: "#0f172a", border: "1px solid #334155" }}
                  labelStyle={{ color: "#e2e8f0" }}
                />
                <Bar dataKey="latency" radius={4}>
                  {data.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
      <p className="mt-3 text-xs text-slate-500">
        Identical weights, different engine. A single call shows little gap; load
        test under concurrency to see ONNX tail-latency advantage.
      </p>
    </div>
  );
}
