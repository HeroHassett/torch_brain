"""Tests for `ignore_intervals` (zero-weighting annotated artifacts) and safe losses."""

import numpy as np
import torch
from temporaldata import Data, Interval, IrregularTimeSeries

from torch_brain.nn import compute_loss_or_metric, prepare_for_multitask_readout
from torch_brain.registry import MODALITIY_REGISTRY, DataType


def _data(config):
    t = np.linspace(0.0, 0.99, 100)
    return Data(
        cursor=IrregularTimeSeries(timestamps=t, vel=np.random.randn(100, 2), domain="auto"),
        cursor_outlier_segments=Interval(start=np.array([0.2, 0.7]), end=np.array([0.3, 0.75])),
        config=config,
        domain=Interval(0.0, 1.0),
    )


def test_ignore_intervals_zero_weights_inside_segments():
    data = _data({"multitask_readout": [{"readout_id": "cursor_velocity_2d",
                                          "ignore_intervals": ["cursor_outlier_segments"]}]})
    ts, _, _, weights, _ = prepare_for_multitask_readout(data, MODALITIY_REGISTRY)
    w = weights["cursor_velocity_2d"]
    ts = np.asarray(ts)
    inside = ((ts >= 0.2) & (ts < 0.3)) | ((ts >= 0.7) & (ts < 0.75))
    assert inside.sum() > 0
    assert np.all(w[inside] == 0) and np.all(w[~inside] == 1)


def test_ignore_intervals_missing_annotation_is_skipped():
    data = _data({"multitask_readout": [{"readout_id": "cursor_velocity_2d",
                                          "ignore_intervals": ["no_such_intervals"]}]})
    _, _, _, weights, _ = prepare_for_multitask_readout(data, MODALITIY_REGISTRY)
    assert np.all(weights["cursor_velocity_2d"] == 1)


def test_weighted_mse_with_all_zero_weights_is_finite():
    out, tgt = torch.randn(10, 2), torch.randn(10, 2)
    loss = compute_loss_or_metric("mse", DataType.CONTINUOUS, out, tgt, torch.zeros(10))
    assert torch.isfinite(loss) and loss.item() == 0.0
    loss = compute_loss_or_metric("mse", DataType.CONTINUOUS, out, tgt, torch.ones(10))
    torch.testing.assert_close(loss, ((out - tgt) ** 2).mean())
