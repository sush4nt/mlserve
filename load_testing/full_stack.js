// k6 smoke/load test that exercises all three endpoints together, useful for
// populating the Grafana dashboard with traffic on every panel.
//
//   k6 run load_testing/full_stack.js

import http from "k6/http";
import { check } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 30 },
    { duration: "60s", target: 30 },
    { duration: "20s", target: 0 },
  ],
  thresholds: { http_req_failed: ["rate < 0.01"] },
};

const BASE = __ENV.BASE_URL || "http://localhost:8080";
const HEADERS = { "Content-Type": "application/json" };

const AVAZU = [14, 2, 0, 1005, 5000, 3000, 3, 3000, 200, 5, 100, 150, 1, 0, 10, 320, 50, 1722, 0, 35, 100, 79];
const TAXI = [1, -73.985, 40.758, -73.978, 40.751, 14, 2, 6, 2014, 0, 0, 1.2, 12.3, 15.1, 8.9, 11.2, 22.4, 25.1];

function body(data, shape, datatype) {
  return JSON.stringify({ inputs: [{ name: "input-0", shape, datatype, data: [data] }] });
}

export default function () {
  const calls = [
    ["avazu-ctr-xgb-py", body(AVAZU, [1, 22], "FP64")],
    ["avazu-ctr-xgb-onnx", body(AVAZU, [1, 22], "FP32")],
    ["nyc-taxi-fare-py", body(TAXI, [1, 18], "FP64")],
  ];
  for (const [model, payload] of calls) {
    const res = http.post(`${BASE}/v2/models/${model}/infer`, payload, { headers: HEADERS });
    check(res, { [`${model} 200`]: (r) => r.status === 200 });
  }
}
