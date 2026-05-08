"""Smoke tests for the typer CLI: subcommands wire up and `--help` works."""

from __future__ import annotations

from typer.testing import CliRunner

from sensorcluster.cli import app

runner = CliRunner()


def test_root_help_lists_all_subcommands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for cmd in ("train", "predict", "evaluate", "serve"):
        assert cmd in out


def test_train_help_runs() -> None:
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "config" in result.stdout.lower()


def test_predict_help_runs() -> None:
    result = runner.invoke(app, ["predict", "--help"])
    assert result.exit_code == 0
    assert "input" in result.stdout.lower()


def test_serve_help_runs() -> None:
    result = runner.invoke(app, ["serve", "--help"])
    assert result.exit_code == 0
    assert "port" in result.stdout.lower()


def test_evaluate_help_runs() -> None:
    result = runner.invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
