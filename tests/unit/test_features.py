"""Preprocessing + dim reduction tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sensorcluster.features.dimreduce import PCAReducer
from sensorcluster.features.preprocess import Preprocessor


@pytest.fixture()
def Xdf() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        rng.normal(size=(50, 5)),
        columns=[f"Sensor {i}" for i in range(5)],
    )


def test_preprocessor_fit_transform_centers_data(Xdf: pd.DataFrame) -> None:
    pre = Preprocessor()
    out = pre.fit_transform(Xdf)
    assert out.shape == Xdf.shape
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-9)


def test_preprocessor_save_load_roundtrip(Xdf: pd.DataFrame, tmp_path: Path) -> None:
    pre = Preprocessor().fit(Xdf)
    expected = pre.transform(Xdf)
    p = tmp_path / "pre.joblib"
    pre.save(p)
    pre2 = Preprocessor.load(p)
    assert pre2.feature_names == list(Xdf.columns)
    assert np.allclose(pre2.transform(Xdf), expected)


def test_preprocessor_reorders_columns_at_transform_time(Xdf: pd.DataFrame) -> None:
    pre = Preprocessor().fit(Xdf)
    shuffled = Xdf[list(reversed(Xdf.columns))]
    out = pre.transform(shuffled)
    assert np.allclose(out, pre.transform(Xdf))


def test_pca_reducer_picks_components_for_variance_target() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10)) @ rng.normal(size=(10, 10))
    pca = PCAReducer(variance_target=0.9).fit(X)
    assert 1 <= pca.n_components_ <= 10
    cum = pca.explained_variance_ratio_.sum()
    assert cum >= 0.9 - 1e-9


def test_pca_reducer_save_load_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 6))
    pca = PCAReducer(variance_target=0.95).fit(X)
    expected = pca.transform(X)
    p = tmp_path / "pca.joblib"
    pca.save(p)
    pca2 = PCAReducer.load(p)
    assert pca2.n_components_ == pca.n_components_
    assert np.allclose(pca2.transform(X), expected)


def test_pca_reducer_rejects_invalid_variance_target() -> None:
    with pytest.raises(ValueError):
        PCAReducer(variance_target=0.0)
    with pytest.raises(ValueError):
        PCAReducer(variance_target=1.5)
