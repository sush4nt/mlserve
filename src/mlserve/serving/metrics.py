"""Prometheus metrics.

Metric names mirror what the original MLServer-based plan expected, so the
Prometheus scrape config and the Grafana dashboard JSON work unchanged:

  rest_server_requests_total          (Counter) method, endpoint, status_code
  rest_server_request_duration_seconds(Histogram) method, endpoint
  rest_server_requests_in_progress    (Gauge) method, endpoint
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

REQUESTS = Counter(
    "rest_server_requests_total",
    "Total inference-server requests.",
    ["method", "endpoint", "status_code"],
)
DURATION = Histogram(
    "rest_server_request_duration_seconds",
    "Request latency in seconds.",
    ["method", "endpoint"],
    buckets=_BUCKETS,
)
IN_PROGRESS = Gauge(
    "rest_server_requests_in_progress",
    "Requests currently being served.",
    ["method", "endpoint"],
)


def render() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
