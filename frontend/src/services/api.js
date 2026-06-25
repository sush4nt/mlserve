import axios from "axios";

// Same-origin in production (FastAPI serves both UI and API); the Vite dev
// server proxies /v2 to the inference server during development.
const BASE = "/v2/models";

/** Build a V2-compliant inference payload from a flat feature array. */
export function buildPayload(featureValues, datatype = "FP64") {
  return {
    inputs: [
      {
        name: "input-0",
        shape: [1, featureValues.length],
        datatype,
        data: [featureValues],
      },
    ],
  };
}

/** Call the V2 infer endpoint. Returns { prediction, latencyMs }. */
export async function infer(modelId, featureValues, datatype = "FP64") {
  const t0 = performance.now();
  const res = await axios.post(
    `${BASE}/${modelId}/infer`,
    buildPayload(featureValues, datatype),
    { headers: { "Content-Type": "application/json" }, timeout: 10000 }
  );
  return {
    prediction: res.data.outputs[0].data[0],
    latencyMs: performance.now() - t0,
  };
}
