"""KServe V2 / Open Inference Protocol — request and response schema.

This is the same contract used by NVIDIA Triton, KServe, and Seldon MLServer.
We implement it directly so the serving layer is a drop-in for any V2 client
(including the k6 load tests and the React frontend) while staying small and
readable.

Spec: https://kserve.github.io/website/modelserving/data_plane/v2_protocol/
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

# V2 datatype string -> numpy dtype. We only need the float types here, but the
# map documents the full set and makes the FP32/FP64 distinction explicit
# (XGBoost runtime accepts FP64; ONNX runtime expects FP32).
V2_DTYPE_TO_NUMPY: dict[str, type] = {
    "FP64": np.float64,
    "FP32": np.float32,
    "INT64": np.int64,
    "INT32": np.int32,
    "BOOL": np.bool_,
}


class InferInput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list  # nested list; flattened against `shape`


class InferRequest(BaseModel):
    inputs: list[InferInput]
    id: str | None = None


class InferOutput(BaseModel):
    name: str
    shape: list[int]
    datatype: str
    data: list


class InferResponse(BaseModel):
    model_name: str = Field(..., alias="model_name")
    id: str | None = None
    outputs: list[InferOutput]

    model_config = {"populate_by_name": True, "protected_namespaces": ()}


def request_to_array(req: InferRequest) -> np.ndarray:
    """Turn the first input tensor of a V2 request into a 2-D numpy array.

    Honors the declared datatype so an ONNX model that wants FP32 actually
    receives FP32. Raises ValueError on an unsupported datatype.
    """
    inp = req.inputs[0]
    dtype = V2_DTYPE_TO_NUMPY.get(inp.datatype)
    if dtype is None:
        raise ValueError(f"Unsupported V2 datatype: {inp.datatype}")
    arr = np.asarray(inp.data, dtype=dtype)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def array_to_response(
    model_name: str, preds: np.ndarray, datatype: str, request_id: str | None
) -> InferResponse:
    """Wrap a prediction array in a V2 response."""
    flat = np.asarray(preds).reshape(-1)
    return InferResponse(
        model_name=model_name,
        id=request_id,
        outputs=[
            InferOutput(
                name="output-0",
                shape=[len(flat), 1],
                datatype=datatype,
                data=flat.astype(float).tolist(),
            )
        ],
    )
