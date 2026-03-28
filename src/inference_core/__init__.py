"""Shared utilities for inference.

Import submodules directly when needed, e.g.:

- ``from src.inference_core.config import build_config``
- ``from src.inference_core.runner_utils import run_prediction_loop``

This package-level module stays intentionally light so simple imports do not
pull in heavy runtime dependencies such as ``torch``.
"""

__all__: list[str] = []
