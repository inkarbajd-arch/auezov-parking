from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class DetectionResult:
    label: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(frozen=True)
class OcrResult:
    text: str
    confidence: float | list[float]


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> list[Any]:
        pass


class BaseOCR(ABC):
    @abstractmethod
    def predict(self, frame: np.ndarray) -> OcrResult:
        pass