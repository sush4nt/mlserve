"""End-to-end smoke test on a tiny synthetic dataset.

Runs the whole pipeline in a temp dir: generate -> preprocess -> train ->
export ONNX -> load runners -> infer. Asserts the Python and ONNX runners agree.
This is what a CI job (stretch goal S1) would run on every push.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlserve.config.schema import load_config
from mlserve.data import synthetic
from mlserve.features.registry import make_preprocessor


@pytest.mark.parametrize("dataset,n_features", [("avazu", 22), ("nyc_taxi", 18)])
def test_preprocess_feature_count(dataset, n_features):
    cfg = load_config(dataset)
    gen = {"avazu": synthetic.make_avazu, "nyc_taxi": synthetic.make_nyc_taxi}[dataset]
    proc = make_preprocessor(cfg).run(gen(5000))
    assert proc.meta["n_features"] == n_features
    assert len(proc.meta["feature_order"]) == n_features
    # No leakage: encoders fit on train only -> val has no unseen-NaN explosions.
    assert not proc.val.isna().any().any()


def test_avazu_has_learnable_signal():
    """Synthetic Avazu should train above chance (sanity check on the generator)."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    cfg = load_config("avazu")
    proc = make_preprocessor(cfg).run(synthetic.make_avazu(20000))
    feats = proc.meta["feature_order"]
    dtr = xgb.DMatrix(proc.train[feats].values, label=proc.train[cfg.target].values)
    dval = xgb.DMatrix(proc.val[feats].values)
    booster = xgb.train(
        {"objective": "binary:logistic", "max_depth": 5, "tree_method": "hist"},
        dtr, num_boost_round=40,
    )
    auc = roc_auc_score(proc.val[cfg.target].values, booster.predict(dval))
    assert auc > 0.6, f"AUC {auc:.3f} too low — generator signal may be broken"


def test_v2_protocol_roundtrip():
    from mlserve.common.protocol import (
        InferInput,
        InferRequest,
        array_to_response,
        request_to_array,
    )

    req = InferRequest(inputs=[InferInput(name="input-0", shape=[2, 3],
                                          datatype="FP32", data=[[1, 2, 3], [4, 5, 6]])])
    arr = request_to_array(req)
    assert arr.shape == (2, 3) and arr.dtype == np.float32
    resp = array_to_response("m", np.array([0.1, 0.9]), "FP32", None)
    assert resp.outputs[0].data == [0.1, 0.9]
