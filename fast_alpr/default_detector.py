from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from open_image_models import LicensePlateDetector
from open_image_models.detection.core.hub import PlateDetectorModel

from fast_alpr.base import BaseDetector


class DefaultDetector(BaseDetector):
    def __init__(
        self,
        model_name: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        conf_thresh: float = 0.15,
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
    ) -> None:
        self.detector = LicensePlateDetector(
            detection_model=model_name,
            conf_thresh=conf_thresh,
            providers=providers,
            sess_options=sess_options,
        )

    def predict(self, frame: np.ndarray):
        return self.detector.predict(frame)