"""Result display panel showing prediction, confidence bar, and history."""

from __future__ import annotations

import tkinter as tk
from typing import List, Optional


class ResultDisplay(tk.Frame):
    """Tkinter panel that displays prediction results with animated bars."""

    HISTORY_SIZE = 5

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg="#0D0D10", **kwargs)
        self._build_ui()
        self._history: List[dict] = []

    # UI construction

    def _build_ui(self) -> None:
        """Create all child widgets."""
        # Title
        tk.Label(
            self,
            text="PREDICTION RESULT",
            bg="#0D0D10",
            fg="#F3F4F6",
            font=("Helvetica", 11, "bold"),
        ).pack(pady=(10, 4))

        # Large digit display
        self._digit_var = tk.StringVar(value="—")
        tk.Label(
            self,
            textvariable=self._digit_var,
            bg="#1C1C21",
            fg="#FFFFFF",
            font=("Helvetica", 96, "bold"),
            width=3,
            relief=tk.RIDGE,
        ).pack(padx=12, pady=6)

        # Sequence string display (for multi-digit uploads)
        self._sequence_var = tk.StringVar(value="")
        tk.Label(
            self,
            textvariable=self._sequence_var,
            bg="#0D0D10",
            fg="#FBBF24",
            font=("Helvetica", 16, "bold"),
        ).pack(pady=(0, 4))

        # Confidence label
        self._conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(
            self,
            textvariable=self._conf_var,
            bg="#0D0D10",
            fg="#34D399",
            font=("Helvetica", 13),
        ).pack()

        # Confidence bar canvas
        self._conf_canvas = tk.Canvas(
            self, bg="#1C1C21", height=18, width=220, highlightthickness=0
        )
        self._conf_canvas.pack(padx=12, pady=(2, 10))

        # Per-class probability bars
        tk.Label(
            self,
            text="All probabilities:",
            bg="#0D0D10",
            fg="#F3F4F6",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, padx=12)

        self._prob_frame = tk.Frame(self, bg="#0D0D10")
        self._prob_frame.pack(fill=tk.X, padx=12, pady=4)
        self._prob_bars: List[tk.Canvas] = []
        self._prob_labels: List[tk.Label] = []
        self._build_prob_bars()

        # History section
        tk.Label(
            self,
            text="Recent predictions:",
            bg="#0D0D10",
            fg="#F3F4F6",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        self._history_frame = tk.Frame(self, bg="#0D0D10")
        self._history_frame.pack(padx=12, pady=2)
        self._history_labels: List[tk.Label] = []
        for _ in range(self.HISTORY_SIZE):
            lbl = tk.Label(
                self._history_frame,
                text="",
                bg="#1C1C21",
                fg="#F3F4F6",
                font=("Helvetica", 18, "bold"),
                width=2,
                relief=tk.RIDGE,
            )
            lbl.pack(side=tk.LEFT, padx=2)
            self._history_labels.append(lbl)

        # Copy button
        tk.Button(
            self,
            text="Copy Result",
            command=self._copy_to_clipboard,
            bg="#3B82F6",
            fg="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(pady=6)

        # Processing time
        self._time_var = tk.StringVar(value="")
        tk.Label(
            self,
            textvariable=self._time_var,
            bg="#0D0D10",
            fg="#9CA3AF",
            font=("Helvetica", 9),
        ).pack(pady=(0, 8))

    def _build_prob_bars(self) -> None:
        """Create 10 small probability bar rows."""
        BAR_W = 140
        BAR_H = 12

        for digit in range(10):
            row = tk.Frame(self._prob_frame, bg="#0D0D10")
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=f"{digit}:",
                bg="#0D0D10",
                fg="#F3F4F6",
                font=("Helvetica", 9),
                width=2,
                anchor=tk.E,
            ).pack(side=tk.LEFT)

            bar = tk.Canvas(
                row, bg="#1C1C21", height=BAR_H, width=BAR_W, highlightthickness=0
            )
            bar.pack(side=tk.LEFT, padx=(3, 4))
            self._prob_bars.append(bar)

            lbl = tk.Label(
                row,
                text="0.0%",
                bg="#0D0D10",
                fg="#F3F4F6",
                font=("Helvetica", 9),
                width=6,
                anchor=tk.W,
            )
            lbl.pack(side=tk.LEFT)
            self._prob_labels.append(lbl)

    # Public API

    def update(self, result) -> None:  # result: PredictionResult
        """Update the display with a new prediction result."""
        self._digit_var.set(str(result.digit))
        self._sequence_var.set("")  # clear sequence when updating single digit
        self._conf_var.set(f"Confidence: {result.confidence * 100:.1f}%")
        self._time_var.set(f"Processing: {result.processing_time_ms:.1f} ms")

        # Confidence bar colour
        conf = result.confidence
        if conf > 0.80:
            bar_color = "#34D399"   # green
        elif conf >= 0.50:
            bar_color = "#FBBF24"   # orange
        else:
            bar_color = "#F87171"   # red

        bar_width = int(conf * 220)
        self._conf_canvas.delete("all")
        if bar_width > 0:
            self._conf_canvas.create_rectangle(
                0, 0, bar_width, 18, fill=bar_color, outline=""
            )
        self._conf_canvas.create_text(
            110, 9, text=f"{conf * 100:.1f}%", fill="#FFFFFF", font=("Helvetica", 9, "bold")
        )

        # Per-class probability bars
        BAR_W = 140
        for digit, prob in enumerate(result.all_probs):
            bar = self._prob_bars[digit]
            bar.delete("all")
            fill_w = int(prob * BAR_W)
            if fill_w > 0:
                color = "#FFFFFF" if digit == result.digit else "#26262B"
                bar.create_rectangle(0, 0, fill_w, 12, fill=color, outline="")
            self._prob_labels[digit].config(text=f"{prob * 100:.1f}%")

        # History
        self._history.append({"digit": result.digit})
        if len(self._history) > self.HISTORY_SIZE:
            self._history.pop(0)
        for i, lbl in enumerate(self._history_labels):
            if i < len(self._history):
                lbl.config(text=str(self._history[i]["digit"]))
            else:
                lbl.config(text="")

    def clear(self) -> None:
        """Reset all displayed values."""
        self._digit_var.set("—")
        self._sequence_var.set("")
        self._conf_var.set("Confidence: —")
        self._time_var.set("")
        self._conf_canvas.delete("all")
        for bar, lbl in zip(self._prob_bars, self._prob_labels):
            bar.delete("all")
            lbl.config(text="0.0%")

    def update_sequence(self, sequence_str: str) -> None:
        """Update the display to show a multi-digit string."""
        self._sequence_var.set(f"Sequence: {sequence_str}")

    # Internal helpers

    def _copy_to_clipboard(self) -> None:
        digit_text = self._digit_var.get()
        if digit_text not in ("—", ""):
            self.clipboard_clear()
            self.clipboard_append(digit_text)
