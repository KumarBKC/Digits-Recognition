"""Webcam live feed panel with digit detection and annotation."""

from __future__ import annotations

import threading
import tkinter as tk
from typing import Optional

import cv2
from PIL import Image, ImageTk
from tkinter import messagebox

from inference.predictor import DigitPredictor
from inference.webcam_stream import WebcamStream


class WebcamPanel(tk.Frame):
    """Tkinter panel that displays a live annotated webcam feed."""

    REFRESH_MS = 30  # ~33 FPS target

    def __init__(
        self,
        parent: tk.Widget,
        predictor: Optional[DigitPredictor] = None,
        on_result=None,
        **kwargs,
    ):
        """
        Args:
            parent: Parent Tkinter widget.
            predictor: Loaded :class:`DigitPredictor` instance.
            on_result: Callback invoked with the highest-confidence
                       :class:`PredictionResult` when a digit is found.
        """
        super().__init__(parent, bg="#0D0D10", **kwargs)
        self._predictor = predictor
        self._on_result = on_result
        self._stream: Optional[WebcamStream] = None
        self._running = False
        self._frame_count = 0
        self._process_every = tk.IntVar(value=1)  # 1 = every frame, 3 = every 3rd
        self._mirror = tk.BooleanVar(value=False)
        self._camera_idx = tk.IntVar(value=0)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._after_id: Optional[str] = None

        self._build_ui()

    # UI construction

    def _build_ui(self) -> None:
        # Controls bar
        controls = tk.Frame(self, bg="#0D0D10")
        controls.pack(fill=tk.X, padx=8, pady=(8, 0))

        # Camera selector
        tk.Label(controls, text="Camera:", bg="#0D0D10", fg="#F3F4F6",
                 font=("Helvetica", 10)).pack(side=tk.LEFT)
        cam_menu = tk.OptionMenu(controls, self._camera_idx, 0, 1, 2, 3)
        cam_menu.config(bg="#26262B", fg="#F3F4F6", relief=tk.FLAT, width=3)
        cam_menu.pack(side=tk.LEFT, padx=(2, 10))

        # Start / Stop button
        self._toggle_btn = tk.Button(
            controls,
            text="▶ Start",
            command=self._toggle,
            bg="#10B981",
            fg="#FFFFFF",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        )
        self._toggle_btn.pack(side=tk.LEFT, padx=4)

        # Processing mode toggle
        tk.Label(controls, text="Process:", bg="#0D0D10", fg="#F3F4F6",
                 font=("Helvetica", 10)).pack(side=tk.LEFT, padx=(10, 2))
        for label, val in (("All frames", 1), ("Every 3rd", 3)):
            tk.Radiobutton(
                controls,
                text=label,
                variable=self._process_every,
                value=val,
                bg="#0D0D10",
                fg="#F3F4F6",
                selectcolor="#26262B",
                activebackground="#0D0D10",
                font=("Helvetica", 10),
            ).pack(side=tk.LEFT)

        # Mirror checkbox
        tk.Checkbutton(
            controls,
            text="Mirror",
            variable=self._mirror,
            bg="#0D0D10",
            fg="#F3F4F6",
            selectcolor="#26262B",
            activebackground="#0D0D10",
            font=("Helvetica", 10),
        ).pack(side=tk.LEFT, padx=(10, 0))

        # Video display label
        self._video_label = tk.Label(
            self, bg="#000000", width=640, height=480
        )
        self._video_label.pack(padx=8, pady=8)

        # FPS label
        self._fps_var = tk.StringVar(value="FPS: —")
        tk.Label(
            self,
            textvariable=self._fps_var,
            bg="#0D0D10",
            fg="#9CA3AF",
            font=("Helvetica", 9),
        ).pack()

    # Stream control

    def _toggle(self) -> None:
        if self._running:
            self._stop()
        else:
            self._start()

    def _start(self) -> None:
        if self._running:
            return
        self._stream = WebcamStream(
            camera_index=self._camera_idx.get(),
            predictor=self._predictor,
        )
        if not self._stream.cap.isOpened():
            self._stream = None
            messagebox.showerror("Camera error", "Could not open camera.")
            return

        self._running = True
        self._toggle_btn.config(text="■ Stop", bg="#EF4444")
        self._frame_count = 0
        self._t0 = __import__("time").perf_counter()
        self._update_frame()

    def _stop(self) -> None:
        self._running = False
        if self._after_id is not None:
            self.after_cancel(self._after_id)
            self._after_id = None
        if self._stream is not None:
            self._stream.release()
            self._stream = None
        self._toggle_btn.config(text="▶ Start", bg="#10B981")
        self._fps_var.set("FPS: —")

    # Frame update loop

    def _update_frame(self) -> None:
        if not self._running or self._stream is None:
            return

        import time

        frame = self._stream.get_frame()
        self._frame_count += 1

        if self._mirror.get():
            frame = cv2.flip(frame, 1)

        # Detection + prediction (throttle)
        results = []
        boxes = []
        if self._predictor is not None and (
            self._frame_count % self._process_every.get() == 0
        ):
            boxes = self._stream.detect_digit_region(frame)
            for x, y, w, h in boxes:
                roi = frame[y : y + h, x : x + w]
                result = self._predictor.predict(roi)
                results.append(result)

            # Notify best result
            if results and self._on_result is not None:
                best = max(results, key=lambda r: r.confidence)
                if self._predictor.is_confident(best):
                    self._on_result(best)

        annotated = self._stream.annotate_frame(frame, boxes, results or [None] * len(boxes))

        # Compute FPS
        elapsed = time.perf_counter() - self._t0
        fps = self._frame_count / max(elapsed, 1e-6)
        self._fps_var.set(f"FPS: {fps:.1f}")

        # Convert to Tkinter image
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img = pil_img.resize((640, 480), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil_img)
        self._video_label.config(image=self._photo)

        self._after_id = self.after(self.REFRESH_MS, self._update_frame)

    # Cleanup

    def destroy(self) -> None:
        self._stop()
        super().destroy()
