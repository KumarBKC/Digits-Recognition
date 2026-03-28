"""Image upload panel with single-digit and multi-digit strip support."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk


class UploadPanel(tk.Frame):
    """Tkinter panel for loading and processing static digit images."""

    ACCEPTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    PREVIEW_W = 400
    PREVIEW_H = 300

    def __init__(
        self,
        parent: tk.Widget,
        on_predict: Optional[Callable] = None,
        on_sequence: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            parent: Parent Tkinter widget.
            on_predict: Callback for single-digit images.
                        Signature: ``callback(image: PIL.Image)``.
            on_sequence: Callback for multi-digit strips.
                         Signature: ``callback(images: list[PIL.Image])``.
        """
        super().__init__(parent, bg="#1e1e2e", **kwargs)
        self._on_predict = on_predict
        self._on_sequence = on_sequence
        self._current_image: Optional[Image.Image] = None
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._build_ui()

    # UI construction

    def _build_ui(self) -> None:
        # Drop zone
        self._drop_zone = tk.Label(
            self,
            text="Drop image here\nor click to browse",
            bg="#313244",
            fg="#585b70",
            font=("Helvetica", 13),
            relief=tk.RIDGE,
            cursor="hand2",
            width=40,
            height=8,
        )
        self._drop_zone.pack(padx=12, pady=(12, 6))
        self._drop_zone.bind("<Button-1>", lambda _: self._browse())

        # Preview area
        self._preview_label = tk.Label(
            self,
            bg="#313244",
            width=self.PREVIEW_W,
            height=self.PREVIEW_H,
        )
        self._preview_label.pack(padx=12, pady=4)

        # Buttons row
        btn_frame = tk.Frame(self, bg="#1e1e2e")
        btn_frame.pack(pady=6)

        tk.Button(
            btn_frame,
            text="Browse…",
            command=self._browse,
            bg="#89b4fa",
            fg="#1e1e2e",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        tk.Button(
            btn_frame,
            text="Detect All Digits",
            command=self._detect_all,
            bg="#a6e3a1",
            fg="#1e1e2e",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.LEFT, padx=4)

        # Status label
        self._status_var = tk.StringVar(value="No image loaded")
        tk.Label(
            self,
            textvariable=self._status_var,
            bg="#1e1e2e",
            fg="#a6adc8",
            font=("Helvetica", 10),
        ).pack(pady=(2, 8))

    # Public API

    def load_image(self, path: str) -> None:
        """Open an image from *path*, display a preview, and run prediction."""
        ext = os.path.splitext(path)[1].lower()
        if ext not in self.ACCEPTED_EXTENSIONS:
            messagebox.showerror("Unsupported format", f"Format '{ext}' is not supported.")
            return

        img = Image.open(path).convert("RGB")
        self._current_image = img
        self._show_preview(img)
        self._status_var.set(f"Loaded: {os.path.basename(path)}")

        # Decide: multi-digit strip or single digit
        w, h = img.size
        if h > 0 and w / h > 2.0:
            self._status_var.set(f"Multi-digit strip detected: {os.path.basename(path)}")
            self._detect_all()
        else:
            self._predict_single(img)

    # Internal helpers

    def _browse(self) -> None:
        """Open a file dialog and load the selected file."""
        filetypes = [
            ("Image files", " ".join(f"*{e}" for e in self.ACCEPTED_EXTENSIONS)),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            self.load_image(path)

    def _show_preview(self, img: Image.Image) -> None:
        """Scale img to fit the preview area and display it."""
        img_copy = img.copy()
        img_copy.thumbnail((self.PREVIEW_W, self.PREVIEW_H), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(img_copy)
        self._preview_label.config(image=self._photo)

    def _predict_single(self, img: Image.Image) -> None:
        """Pass image to the single-predict callback."""
        if self._on_predict is not None:
            self._on_predict(img)

    def _detect_all(self) -> None:
        """Segment multi-digit strip and predict each digit."""
        if self._current_image is None:
            return

        rois = self._segment_digits(self._current_image)
        if not rois:
            self._status_var.set("No digits detected in image.")
            return

        self._status_var.set(f"Detected {len(rois)} digit region(s)")

        if self._on_sequence is not None:
            self._on_sequence(rois)

    @staticmethod
    def _segment_digits(img: Image.Image) -> List[Image.Image]:
        """Segment a multi-digit strip into individual digit ROI images.

        Returns list of PIL images, sorted left-to-right.
        """
        gray_np = np.array(img.convert("L"))
        _, binary = cv2.threshold(gray_np, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours:
            if cv2.contourArea(c) < 50:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        # Sort left-to-right
        boxes.sort(key=lambda b: b[0])

        rois: List[Image.Image] = []
        for x, y, w, h in boxes:
            pad = 4
            h_img, w_img = gray_np.shape
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w_img, x + w + pad), min(h_img, y + h + pad)
            roi_np = gray_np[y0:y1, x0:x1]
            rois.append(Image.fromarray(roi_np))

        return rois
