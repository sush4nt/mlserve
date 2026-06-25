import { useState } from "react";
import config from "./config/models.generated.json";
import PredictionForm from "./components/PredictionForm.jsx";
import ComparisonPanel from "./components/ComparisonPanel.jsx";

const MODELS = config.models;
const BADGE = {
  python: "bg-blue-500/20 text-blue-300",
  onnx: "bg-green-500/20 text-green-300",
};

export default function App() {
  // Default to the first Python endpoint.
  const [selectedId, setSelectedId] = useState(
    (MODELS.find((m) => m.runtime === "python") || MODELS[0])?.id
  );
  const model = MODELS.find((m) => m.id === selectedId);

  const avazuPy = MODELS.find((m) => m.id === "avazu-ctr-xgb-py");
  const avazuOnnx = MODELS.find((m) => m.id === "avazu-ctr-xgb-onnx");
  const showComparison = model?.dataset === "avazu" && avazuPy && avazuOnnx;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-xl font-bold">
          mlserve <span className="text-slate-500">— inference platform</span>
        </h1>
        <p className="text-sm text-slate-500">
          XGBoost vs ONNX Runtime · KServe V2 protocol · FastAPI + Prometheus
        </p>
      </header>

      <main className="mx-auto max-w-3xl space-y-6 px-6 py-8">
        {/* Model selector */}
        <div className="flex flex-wrap gap-2">
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedId(m.id)}
              className={`rounded-lg px-3 py-2 text-sm ring-1 ring-slate-700 ${
                m.id === selectedId ? "bg-slate-800 text-white" : "bg-slate-900 text-slate-400"
              }`}
            >
              {m.displayName}
            </button>
          ))}
        </div>

        {model && (
          <section className="space-y-4">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold">{model.displayName}</h2>
              <span className={`rounded px-2 py-0.5 text-xs ${BADGE[model.runtimeBadge] || ""}`}>
                {model.runtime}
              </span>
              <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
                {model.datatype}
              </span>
            </div>
            <p className="text-sm text-slate-400">{model.description}</p>

            {model.fields.some((f) => f.editable) ? (
              <PredictionForm model={model} />
            ) : (
              <p className="rounded-lg bg-slate-800 p-4 text-sm text-slate-400">
                This endpoint shares inputs with its Python peer — use the
                comparison panel below.
              </p>
            )}
          </section>
        )}

        {showComparison && <ComparisonPanel pyModel={avazuPy} onnxModel={avazuOnnx} />}
      </main>
    </div>
  );
}
