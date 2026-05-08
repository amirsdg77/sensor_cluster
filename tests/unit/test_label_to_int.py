"""Tests for `_label_to_int`: encodes label names for ARI computation."""

from __future__ import annotations

import pytest

from sensorcluster.pipeline.train import _label_to_int


def test_class_label_encodes_to_class_number() -> None:
    assert _label_to_int("CLASS_1") == 1
    assert _label_to_int("CLASS_42") == 42


def test_unknown_label_encodes_above_class_range() -> None:
    assert _label_to_int("UNKNOWN_0") == 1000
    assert _label_to_int("UNKNOWN_7") == 1007


def test_noise_label_encodes_to_minus_one() -> None:
    assert _label_to_int("NOISE") == -1


def test_malformed_class_name_raises() -> None:
    with pytest.raises(ValueError, match="CLASS_"):
        _label_to_int("CLASS_not_an_int")


def test_malformed_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="UNKNOWN_"):
        _label_to_int("UNKNOWN_not_an_int")


def test_unrecognized_pattern_raises() -> None:
    with pytest.raises(ValueError, match="Unrecognized"):
        _label_to_int("SOMETHING_ELSE")
