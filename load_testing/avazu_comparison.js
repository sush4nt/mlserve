// k6 load test: hammer the Avazu Python and ONNX endpoints with the same input
// and track per-runtime latency. The spike stage is where ONNX's tail-latency
// advantage shows up — a single request barely differs.
//
//   k6 run load_testing/avazu_comparison.js
//   k6 run --out json=load_testing/results/avazu.json load_testing/avazu_comparison.js
//
// Set BASE_URL for a deployed target, e.g.:
//   k6 run -e BASE_URL=https://<user>-mlserve.hf.space load_testing/avazu_comparison.js

import http from "k6/http";
import { check, group } from "k6";
import { Trend } from "k6/metrics";

const pyLatency = new Trend("py_inference_ms", true);
const onnxLatency = new Trend("onnx_inference_ms", true);

export const options = {
  stages: [
    { duration: "20s", target: 20 }, // ramp up
    { duration: "40s", target: 20 }, // baseline
    { duration: "20s", target: 80 }, // spike
    { duration: "40s", target: 80 }, // sustain under load
    { duration: "20s", target: 0 },  // ramp down
  ],
  thresholds: {
    py_inference_ms: ["p(99) < 500"],
    onnx_inference_ms: ["p(99) < 300"],
    http_req_failed: ["rate < 0.01"],
  },
};

const BASE = __ENV.BASE_URL || "http://localhost:8080";
const PY_URL = `${BASE}/v2/models/avazu-ctr-xgb-py/infer`;
const ONNX_URL = `${BASE}/v2/models/avazu-ctr-xgb-onnx/infer`;
const HEADERS = { "Content-Type": "application/json" };

// 22 features in the canonical order from feature_meta.json.
const SAMPLE = [14, 2, 0, 1005, 5000, 3000, 3, 3000, 200, 5, 100, 150, 1, 0, 10, 320, 50, 1722, 0, 35, 100, 79];

function payload(datatype) {
  return JSON.stringify({
    inputs: [{ name: "input-0", shape: [1, 22], datatype, data: [SAMPLE] }],
  });
}

export default function () {
  group("Python XGBoost", () => {
    const res = http.post(PY_URL, payload("FP64"), { headers: HEADERS });
    check(res, { "py 200": (r) => r.status === 200 });
    pyLatency.add(res.timings.duration);
  });

  group("ONNX Runtime", () => {
    const res = http.post(ONNX_URL, payload("FP32"), { headers: HEADERS });
    check(res, { "onnx 200": (r) => r.status === 200 });
    onnxLatency.add(res.timings.duration);
  });
}
