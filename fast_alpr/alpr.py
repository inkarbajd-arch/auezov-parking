import os
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from fast_alpr.base import BaseDetector, BaseOCR, OcrResult
from fast_alpr.default_detector import DefaultDetector
from fast_alpr.default_ocr import DefaultOCR


@dataclass(frozen=True)
class ALPRResult:
    detection: object
    ocr: OcrResult | None


@dataclass(frozen=True, slots=True)
class DrawPredictionsResult:
    image: np.ndarray
    results: list[ALPRResult]


class ALPR:
    def __init__(
        self,
        detector: BaseDetector | None = None,
        ocr: BaseOCR | None = None,
        detector_model: PlateDetectorModel = "yolo-v9-t-384-license-plate-end2end",
        detector_conf_thresh: float = 0.15,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions | None = None,
        ocr_model: OcrModel | None = "cct-xs-v2-global-model",
        ocr_device: Literal["cuda", "cpu", "auto"] = "cpu",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
        ocr_model_path: str | os.PathLike | None = None,
        ocr_config_path: str | os.PathLike | None = None,
        ocr_force_download: bool = False,
    ) -> None:
        self.detector = detector or DefaultDetector(
            model_name=detector_model,
            conf_thresh=detector_conf_thresh,
            providers=detector_providers,
            sess_options=detector_sess_options,
        )

        self.ocr = ocr or DefaultOCR(
            hub_ocr_model=ocr_model,
            device=ocr_device,
            providers=ocr_providers,
            sess_options=ocr_sess_options,
            model_path=ocr_model_path,
            config_path=ocr_config_path,
            force_download=ocr_force_download,
        )

    def predict(self, frame: np.ndarray | str) -> list[ALPRResult]:
        img = cv2.imread(frame) if isinstance(frame, str) else frame

        if img is None:
            raise ValueError("Image/frame not found")

        detections = self.detector.predict(img)
        results: list[ALPRResult] = []

        for detection in detections:
            bbox = self.get_bbox(detection)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            cropped = img[y1:y2, x1:x2]

            ocr_result = None
            if cropped.size > 0:
                ocr_result = self.ocr.predict(cropped)

            results.append(ALPRResult(detection=detection, ocr=ocr_result))

        return results

    def best_plate(self, frame: np.ndarray) -> str | None:
        results = self.predict(frame)

        best_text = None
        best_conf = 0.0

        for result in results:
            if not result.ocr or not result.ocr.text:
                continue

            conf = result.ocr.confidence

            if isinstance(conf, list):
                conf = statistics.mean(conf) if conf else 0.0

            conf = float(conf or 0.0)

            if conf >= best_conf:
                best_conf = conf
                best_text = result.ocr.text

        return best_text

    def draw_predictions(self, frame: np.ndarray | str) -> DrawPredictionsResult:
        img = cv2.imread(frame) if isinstance(frame, str) else frame.copy()

        if img is None:
            raise ValueError("Image/frame not found")

        results = self.predict(img)

        for result in results:
            bbox = self.get_bbox(result.detection)

            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox

            cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)

            if result.ocr and result.ocr.text:
                conf = result.ocr.confidence

                if isinstance(conf, list):
                    conf = statistics.mean(conf) if conf else 0.0

                label = f"{result.ocr.text} ({float(conf or 0) * 100:.0f}%)"

                cv2.putText(
                    img,
                    label,
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (36, 255, 12),
                    2,
                )

        return DrawPredictionsResult(image=img, results=results)

    @staticmethod
    def get_bbox(detection):
        try:
            box = detection.bounding_box
            x1 = int(box.x1)
            y1 = int(box.y1)
            x2 = int(box.x2)
            y2 = int(box.y2)
            return max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
        except Exception:
            pass

        try:
            box = detection["bounding_box"]
            return (
                int(box["x1"]),
                int(box["y1"]),
                int(box["x2"]),
                int(box["y2"]),
            )
        except Exception:
            pass

        try:
            x1, y1, x2, y2 = detection[:4]
            return int(x1), int(y1), int(x2), int(y2)
        except Exception:
            return None