// Renders a classification probability as a bar, or a regression value as text.
export default function ResultDisplay({ model, result }) {
  if (result == null) return null;

  if (model.outputType === "probability") {
    const pct = Math.max(0, Math.min(1, result.prediction)) * 100;
    return (
      <div className="mt-4">
        <div className="flex justify-between text-sm text-slate-400">
          <span>{model.outputLabel}</span>
          <span className="font-mono text-slate-200">{result.prediction.toFixed(4)}</span>
        </div>
        <div className="mt-1 h-3 w-full rounded bg-slate-700">
          <div
            className="h-3 rounded bg-blue-500 transition-all"
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4">
      <div className="text-sm text-slate-400">{model.outputLabel}</div>
      <div className="font-mono text-3xl text-emerald-400">
        ${result.prediction.toFixed(2)}
      </div>
    </div>
  );
}
