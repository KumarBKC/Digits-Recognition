"""Result display panel showing prediction, confidence bar, and history."""

from __future__ import annotations

import tkinter as tk
from typing import List, Optional


class ResultDisplay(tk.Frame):
    """Tkinter panel that displays prediction results with animated bars."""

    HISTORY_SIZE = 5

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg="#1e1e2e", **kwargs)
        self._build_ui()
        self._history: List[dict] = []

    # UI construction

    def _build_ui(self) -> None:
        """Create all child widgets."""
        # Title
        tk.Label(
            self,
            text="PREDICTION RESULT",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 11, "bold"),
        ).pack(pady=(10, 4))

        # Large digit display
        self._digit_var = tk.StringVar(value="—")
        tk.Label(
            self,
            textvariable=self._digit_var,
            bg="#313244",
            fg="#cba6f7",
            font=("Helvetica", 96, "bold"),
            width=3,
            relief=tk.RIDGE,
        ).pack(padx=12, pady=6)

        # Confidence label
        self._conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(
            self,
            textvariable=self._conf_var,
            bg="#1e1e2e",
            fg="#a6e3a1",
            font=("Helvetica", 13),
        ).pack()

        # Confidence bar canvas
        self._conf_canvas = tk.Canvas(
            self, bg="#313244", height=18, width=220, highlightthickness=0
        )
        self._conf_canvas.pack(padx=12, pady=(2, 10))

        # Per-class probability bars
        tk.Label(
            self,
            text="All probabilities:",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, padx=12)

        self._prob_frame = tk.Frame(self, bg="#1e1e2e")
        self._prob_frame.pack(fill=tk.X, padx=12, pady=4)
        self._prob_bars: List[tk.Canvas] = []
        self._prob_labels: List[tk.Label] = []
        self._build_prob_bars()

        # History section
        tk.Label(
            self,
            text="Recent predictions:",
            bg="#1e1e2e",
            fg="#cdd6f4",
            font=("Helvetica", 10),
        ).pack(anchor=tk.W, padx=12, pady=(8, 2))

        self._history_frame = tk.Frame(self, bg="#1e1e2e")
        self._history_frame.pack(padx=12, pady=2)
        self._history_labels: List[tk.Label] = []
        for _ in range(self.HISTORY_SIZE):
            lbl = tk.Label(
                self._history_frame,
                text="",
                bg="#313244",
                fg="#cdd6f4",
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
            bg="#45475a",
            fg="#cdd6f4",
            relief=tk.FLAT,
            cursor="hand2",
        ).pack(pady=6)

        # Processing time
        self._time_var = tk.StringVar(value="")
        tk.Label(
            self,
            textvariable=self._time_var,
            bg="#1e1e2e",
            fg="#585b70",
            font=("Helvetica", 9),
        ).pack(pady=(0, 8))

    def _build_prob_bars(self) -> None:
        """Create 10 small probability bar rows."""
        BAR_W = 140
        BAR_H = 12

        for digit in range(10):
            row = tk.Frame(self._prob_frame, bg="#1e1e2e")
            row.pack(fill=tk.X, pady=1)

            tk.Label(
                row,
                text=f"{digit}:",
                bg="#1e1e2e",
                fg="#cdd6f4",
                font=("Helvetica", 9),
                width=2,
                anchor=tk.E,
            ).pack(side=tk.LEFT)

            bar = tk.Canvas(
                row, bg="#313244", height=BAR_H, width=BAR_W, highlightthickness=0
            )
            bar.pack(side=tk.LEFT, padx=(3, 4))
            self._prob_bars.append(bar)

            lbl = tk.Label(
                row,
                text="0.0%",
                bg="#1e1e2e",
                fg="#cdd6f4",
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
        self._conf_var.set(f"Confidence: {result.confidence * 100:.1f}%")
        self._time_var.set(f"Processing: {result.processing_time_ms:.1f} ms")

        # Confidence bar colour
        conf = result.confidence
        if conf > 0.80:
            bar_color = "#a6e3a1"   # green
        elif conf >= 0.50:
            bar_color = "#fab387"   # orange
        else:
            bar_color = "#f38ba8"   # red

        bar_width = int(conf * 220)
        self._conf_canvas.delete("all")
        if bar_width > 0:
            self._conf_canvas.create_rectangle(
                0, 0, bar_width, 18, fill=bar_color, outline=""
            )
        self._conf_canvas.create_text(
            110, 9, text=f"{conf * 100:.1f}%", fill="#1e1e2e", font=("Helvetica", 9, "bold")
        )

        # Per-class probability bars
        BAR_W = 140
        for digit, prob in enumerate(result.all_probs):
            bar = self._prob_bars[digit]
            bar.delete("all")
            fill_w = int(prob * BAR_W)
            if fill_w > 0:
                color = "#cba6f7" if digit == result.digit else "#45475a"
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
        self._conf_var.set("Confidence: —")
        self._time_var.set("")
        self._conf_canvas.delete("all")
        for bar, lbl in zip(self._prob_bars, self._prob_labels):
            bar.delete("all")
            lbl.config(text="0.0%")

    # Internal helpers

    def _copy_to_clipboard(self) -> None:
        digit_text = self._digit_var.get()
        if digit_text not in ("—", ""):
            self.clipboard_clear()
            self.clipboard_append(digit_text)
