from collections.abc import Sequence

import numpy as np
import onnxruntime as ort
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference.hub import OcrModel

from fast_alpr.base import BaseOCR, OcrResult


class DefaultOCR(BaseOCR):
    def __init__(
        self,
        hub_ocr_model: OcrModel | None = "cct-xs-v2-global-model",
        device: str = "cpu",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        model_path: str | None = None,
        config_path: str | None = None,
        force_download: bool = False,
    ) -> None:
        try:
            self.ocr = LicensePlateRecognizer(
                hub_ocr_model=hub_ocr_model,
                device=device,
                providers=providers,
                sess_options=sess_options,
            )
        except TypeError:
            self.ocr = LicensePlateRecognizer(
                hub_ocr_model=hub_ocr_model,
                device=device,
            )

    def predict(self, frame: np.ndarray) -> OcrResult:
        try:
            result = self.ocr.run(frame)

            if isinstance(result, list) and result:
                item = result[0]
            else:
                item = result

            text = getattr(item, "text", "")
            confidence = getattr(item, "confidence", 0.0)

            if not text:
                text = str(item)

            return OcrResult(
                text=self.clean_plate(text),
                confidence=confidence or 0.0,
            )

        except Exception as e:
            print("OCR error:", e)
            return OcrResult(text="", confidence=0.0)

    @staticmethod
    def clean_plate(text: str) -> str:
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = text.upper().replace(" ", "").replace("-", "").replace("_", "")
        return "".join(ch for ch in text if ch in allowed)