"""Main application window — Tkinter desktop GUI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import tkinter as tk
from tkinter import messagebox
from typing import Optional

# Ensure package root is on the path when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from inference.predictor import DigitPredictor, PredictionResult
from ui.canvas_panel import CanvasPanel
from ui.result_display import ResultDisplay
from ui.upload_panel import UploadPanel
from ui.webcam_panel import WebcamPanel

_CONFIG_FILE = os.path.join(os.path.expanduser("~"), ".digit_recognition_config.json")

MODE_WEBCAM = "WEBCAM"
MODE_UPLOAD = "UPLOAD"
MODE_DRAW = "DRAW"


def _load_config() -> dict:
    if os.path.exists(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_mode": MODE_DRAW}


def _save_config(cfg: dict) -> None:
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    except Exception:
        pass


class MainApp(tk.Tk):
    """Main application window for the Handwritten Digit Recognition System."""

    MIN_W = 1000
    MIN_H = 600

    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.title("Digit Recognition System")
        self.minsize(self.MIN_W, self.MIN_H)
        self.configure(bg="#0D0D10")

        self._model_path = model_path
        self._predictor: Optional[DigitPredictor] = None
        self._config = _load_config()
        self._current_mode: str = self._config.get("last_mode", MODE_DRAW)

        self._build_menu_bar()
        self._build_header()
        self._build_main_area()
        self._build_status_bar()

        self._bind_shortcuts()

        # Load model in background thread
        self._loading_label: Optional[tk.Label] = None
        self._show_loading()
        threading.Thread(target=self._load_model, daemon=True).start()

    # UI Construction

    def _build_menu_bar(self) -> None:
        """Build a minimal menu bar."""
        menubar = tk.Menu(self, bg="#1C1C21", fg="#F3F4F6")
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=False, bg="#1C1C21", fg="#F3F4F6")
        file_menu.add_command(label="Open image…", command=self._open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menubar.add_cascade(label="File", menu=file_menu)

        view_menu = tk.Menu(menubar, tearoff=False, bg="#1C1C21", fg="#F3F4F6")
        view_menu.add_command(label="Webcam (Ctrl+1)", command=lambda: self._switch_mode(MODE_WEBCAM))
        view_menu.add_command(label="Upload (Ctrl+2)", command=lambda: self._switch_mode(MODE_UPLOAD))
        view_menu.add_command(label="Draw (Ctrl+3)", command=lambda: self._switch_mode(MODE_DRAW))
        menubar.add_cascade(label="View", menu=view_menu)

    def _build_header(self) -> None:
        """Build the top navigation bar with mode buttons."""
        header = tk.Frame(self, bg="#000000", pady=6)
        header.pack(fill=tk.X)

        tk.Label(
            header,
            text="Digit Recognition System",
            bg="#000000",
            fg="#FFFFFF",
            font=("Helvetica", 14, "bold"),
        ).pack(side=tk.LEFT, padx=12)

        btn_frame = tk.Frame(header, bg="#000000")
        btn_frame.pack(side=tk.RIGHT, padx=8)

        self._mode_buttons: dict[str, tk.Button] = {}
        for label, mode in (("● Live", MODE_WEBCAM), ("↑ Upload", MODE_UPLOAD), ("✎ Draw", MODE_DRAW)):
            btn = tk.Button(
                btn_frame,
                text=label,
                command=lambda m=mode: self._switch_mode(m),
                bg="#1F2937",
                fg="#F3F4F6",
                relief=tk.FLAT,
                font=("Helvetica", 10, "bold"),
                cursor="hand2",
                padx=10,
                pady=4,
            )
            btn.pack(side=tk.LEFT, padx=3)
            self._mode_buttons[mode] = btn

    def _build_main_area(self) -> None:
        """Build the two-column main area (active panel + result display)."""
        main = tk.Frame(self, bg="#0D0D10")
        main.pack(fill=tk.BOTH, expand=True)

        # Left: active panel area
        self._panel_frame = tk.Frame(main, bg="#1C1C21")
        self._panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: result display
        self._result_display = ResultDisplay(main, width=280)
        self._result_display.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 0))

        # Create all panels (hidden initially)
        self._webcam_panel = WebcamPanel(
            self._panel_frame,
            predictor=None,  # set later after model loads
            on_result=self._on_prediction,
        )
        self._upload_panel = UploadPanel(
            self._panel_frame,
            on_predict=self._predict_image,
            on_sequence=self._predict_sequence,
        )
        self._canvas_panel = CanvasPanel(
            self._panel_frame,
            on_predict=self._predict_image,
            on_clear=self._result_display.clear,
        )

        # Show last-used mode
        self._active_panel: Optional[tk.Frame] = None
        self._switch_mode(self._current_mode, save=False)

    def _build_status_bar(self) -> None:
        """Build the bottom status bar."""
        status_bar = tk.Frame(self, bg="#000000", pady=4)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._status_device_var = tk.StringVar(value="Device: —")
        self._status_model_var = tk.StringVar(value="Model: not loaded")

        for var in (self._status_device_var, self._status_model_var):
            tk.Label(
                status_bar,
                textvariable=var,
                bg="#000000",
                fg="#9CA3AF",
                font=("Helvetica", 9),
            ).pack(side=tk.LEFT, padx=12)

    # Mode switching

    def _switch_mode(self, mode: str, save: bool = True) -> None:
        """Show the panel for *mode* and hide all others."""
        # Stop webcam if leaving webcam mode
        if self._current_mode == MODE_WEBCAM and mode != MODE_WEBCAM:
            self._webcam_panel._stop()

        self._current_mode = mode
        if save:
            self._config["last_mode"] = mode
            _save_config(self._config)

        # Update button highlights
        for m, btn in self._mode_buttons.items():
            btn.config(bg="#3B82F6" if m == mode else "#1F2937")

        # Hide current panel and show new one
        if self._active_panel is not None:
            self._active_panel.pack_forget()

        panel_map = {
            MODE_WEBCAM: self._webcam_panel,
            MODE_UPLOAD: self._upload_panel,
            MODE_DRAW: self._canvas_panel,
        }
        self._active_panel = panel_map.get(mode)
        if self._active_panel is not None:
            self._active_panel.pack(fill=tk.BOTH, expand=True)

        self._result_display.clear()

    # Keyboard shortcuts

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-Key-1>", lambda _: self._switch_mode(MODE_WEBCAM))
        self.bind("<Control-Key-2>", lambda _: self._switch_mode(MODE_UPLOAD))
        self.bind("<Control-Key-3>", lambda _: self._switch_mode(MODE_DRAW))

    # Model loading

    def _show_loading(self) -> None:
        self._loading_label = tk.Label(
            self._panel_frame,
            text="Loading model…",
            bg="#1C1C21",
            fg="#F3F4F6",
            font=("Helvetica", 18),
        )
        self._loading_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def _load_model(self) -> None:
        """Load DigitPredictor in background; update UI when done."""
        try:
            if self._model_path and os.path.exists(self._model_path):
                predictor = DigitPredictor(self._model_path)
                self._predictor = predictor
                # Inject into webcam panel
                self._webcam_panel._predictor = predictor
                model_name = os.path.basename(self._model_path)
                self.after(0, lambda: self._status_model_var.set(f"Model: {model_name}"))
                import torch
                device_str = "CUDA" if torch.cuda.is_available() else "CPU"
                self.after(0, lambda: self._status_device_var.set(f"Device: {device_str}"))
            else:
                self.after(
                    0,
                    lambda: self._status_model_var.set("Model: not found — train first"),
                )
        except Exception as exc:
            self.after(
                0,
                lambda: self._status_model_var.set(f"Model error: {exc}"),
            )
        finally:
            if self._loading_label is not None:
                self.after(0, self._loading_label.destroy)

    # Prediction callbacks

    def _predict_image(self, image) -> None:
        """Predict a single image (PIL or ndarray) and update the display."""
        if self._predictor is None:
            messagebox.showwarning("No model", "Please load a trained model first.")
            return
        result = self._predictor.predict(image)
        self._on_prediction(result)

    def _predict_sequence(self, images: list) -> None:
        """Predict a sequence of images and show combined result."""
        if self._predictor is None:
            return
        results = self._predictor.predict_batch(images)
        digits = "".join(str(r.digit) for r in results)
        # Show best-confidence individual result in the panel
        if results:
            best = max(results, key=lambda r: r.confidence)
            self._on_prediction(best)
        # TODO: could display full sequence string in a separate label

    def _on_prediction(self, result: PredictionResult) -> None:
        """Update the result display from any thread."""
        self.after(0, lambda: self._result_display.update(result))

    # File menu helpers

    def _open_image(self) -> None:
        self._switch_mode(MODE_UPLOAD)
        self._upload_panel._browse()


# Entry point

def main() -> None:
    parser = argparse.ArgumentParser(description="Digit Recognition UI")
    parser.add_argument(
        "--model",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to trained checkpoint (.pth).",
    )
    args = parser.parse_args()

    app = MainApp(model_path=args.model)
    app.mainloop()


if __name__ == "__main__":
    main()
