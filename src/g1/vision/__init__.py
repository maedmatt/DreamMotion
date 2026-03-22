from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = ["Detection", "ObjectDetector", "get_detector"]

if TYPE_CHECKING:
    from g1.vision.detector import Detection, ObjectDetector, get_detector


def __getattr__(name: str):
    if name in __all__:
        module = import_module("g1.vision.detector")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
