"""Pydantic API schema tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sensorcluster.api.schemas import (
    PredictBatchRequest,
    PredictRequest,
    PredictResponse,
)


def test_predict_request_accepts_valid_vector() -> None:
    req = PredictRequest(sensors=[0.1, -0.2, 0.0])
    assert req.sensors == [0.1, -0.2, 0.0]


def test_predict_request_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sensors=[float("nan")])


def test_predict_request_rejects_inf() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sensors=[float("inf")])


def test_predict_request_rejects_empty_vector() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sensors=[])


def test_predict_response_clamps_via_validation() -> None:
    with pytest.raises(ValidationError):
        PredictResponse(
            predicted_label="X",
            confidence=1.5,  # out of [0, 1]
            novelty_score=0.5,
            cluster_id=0,
            is_novel=False,
            top_neighbors=[],
        )


def test_predict_batch_request_rejects_empty_rows() -> None:
    with pytest.raises(ValidationError):
        PredictBatchRequest(rows=[])


def test_predict_request_rejects_value_above_band() -> None:
    with pytest.raises(ValidationError, match="must be in"):
        PredictRequest(sensors=[1.5])


def test_predict_request_rejects_value_below_band() -> None:
    with pytest.raises(ValidationError, match="must be in"):
        PredictRequest(sensors=[-1.5])


def test_predict_request_rejects_oversized_vector() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(sensors=[0.0] * 2000)


def test_predict_batch_request_rejects_out_of_band_value() -> None:
    with pytest.raises(ValidationError, match="outside"):
        PredictBatchRequest(rows=[[0.1, 0.2, 9.0]])
