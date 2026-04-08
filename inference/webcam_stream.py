"""Real-time webcam capture and digit-region detection."""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from inference.predictor import DigitPredictor, PredictionResult

# Type alias for bounding boxes: (x, y, w, h)
BoundingBox = Tuple[int, int, int, int]


def _iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute intersection-over-union of two (x, y, w, h) boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ix = max(ax, bx)
    iy = max(ay, by)
    iw = min(ax + aw, bx + bw) - ix
    ih = min(ay + ah, by + bh) - iy

    if iw <= 0 or ih <= 0:
        return 0.0

    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _nms(boxes: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
    """Non-maximum suppression — keep the larger box when IoU > threshold."""
    if not boxes:
        return boxes

    # Sort by area descending (keep larger box)
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    kept: List[BoundingBox] = []
    suppressed = [False] * len(boxes)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        kept.append(boxes[i])
        for j in range(i + 1, len(boxes)):
            if not suppressed[j] and _iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed[j] = True

    return kept


class WebcamStream:
    """OpenCV webcam capture with digit-ROI detection and annotation."""

    def __init__(
        self,
        camera_index: int = 0,
        predictor: Optional[DigitPredictor] = None,
    ):
        self.predictor = predictor
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Frame capture

    def get_frame(self) -> np.ndarray:
        """Return the latest BGR frame; black placeholder on failure."""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    # ROI detection

    def detect_digit_region(self, frame: np.ndarray) -> List[BoundingBox]:
        """Detect digit bounding boxes in a BGR frame.

        Steps:
          1. Grayscale conversion
          2. Gaussian blur
          3. Adaptive threshold
          4. Morphological closing (fill stroke gaps)
          5. Contour finding and filtering
          6. Non-maximum suppression

        Returns:
            List of ``(x, y, w, h)`` bounding rectangles.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        h_frame, w_frame = frame.shape[:2]
        edge_pad = 5
        boxes: List[BoundingBox] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 300:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # Aspect ratio filter: digits are generally taller than wide
            if h == 0 or not (0.1 < w / h < 1.5):
                continue

            # Skip boxes touching frame edges
            if (
                x < edge_pad
                or y < edge_pad
                or x + w > w_frame - edge_pad
                or y + h > h_frame - edge_pad
            ):
                continue

            boxes.append((x, y, w, h))

        return _nms(boxes, iou_threshold=0.5)

    # Frame annotation

    def annotate_frame(
        self,
        frame: np.ndarray,
        boxes: List[BoundingBox],
        results: List[Optional[PredictionResult]],
    ) -> np.ndarray:
        """Draw bounding boxes and prediction labels on a copy of the frame.

        Args:
            frame: Original BGR frame.
            boxes: List of ``(x, y, w, h)`` regions.
            results: Prediction result for each box (``None`` = skip label).

        Returns:
            Annotated BGR frame.
        """
        annotated = frame.copy()

        for box, result in zip(boxes, results):
            x, y, w, h = box

            # Colour coding by confidence
            if result is None:
                color = (128, 128, 128)
                label = "?"
            else:
                conf = result.confidence
                if conf > 0.80:
                    color = (0, 200, 0)   # green
                elif conf >= 0.65:
                    color = (0, 165, 255)  # orange
                else:
                    color = (0, 0, 255)   # red
                label = f"{result.digit} - {conf * 100:.1f}%"

            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

            # Label above box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x
            text_y = max(y - 8, text_size[1] + 4)
            cv2.putText(
                annotated,
                label,
                (text_x, text_y),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    # Cleanup

    def release(self) -> None:
        """Release the camera resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()
