"""sensorcluster — semi-supervised sensor clustering with HDBSCAN."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sensorcluster")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
