"""Drawing canvas panel for on-screen digit input."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Callable, Optional

from PIL import Image, ImageDraw

class CanvasPanel(tk.Frame):
    """Tkinter panel with a drawing canvas for mouse/touch digit input."""

    CANVAS_SIZE = 280

    def __init__(
        self,
        parent: tk.Widget,
        on_predict: Optional[Callable] = None,
        on_clear: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Args:
            parent: Parent Tkinter widget.
            on_predict: Callable invoked with a PIL.Image when recognition
                        is triggered.  Signature: ``callback(image: PIL.Image)``.
            on_clear: Callable invoked when the canvas is cleared or empty.
        """
        super().__init__(parent, bg="#0D0D10", **kwargs)
        self._on_predict = on_predict
        self._on_clear = on_clear
        self._auto_recognize = tk.BooleanVar(value=True)
        self._brush_size = tk.IntVar(value=14)
        self._prev_x: Optional[int] = None
        self._prev_y: Optional[int] = None

        # PIL image backing the canvas for pixel-accurate export
        self._pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 255)
        self._pil_draw = ImageDraw.Draw(self._pil_image)

        self._build_ui()

    # UI construction

    def _build_ui(self) -> None:
        # Controls row
        controls = tk.Frame(self, bg="#0D0D10")
        controls.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Label(
            controls, text="Brush:", bg="#0D0D10", fg="#F3F4F6", font=("Helvetica", 10)
        ).pack(side=tk.LEFT)

        tk.Scale(
            controls,
            variable=self._brush_size,
            from_=8,
            to=24,
            orient=tk.HORIZONTAL,
            bg="#0D0D10",
            fg="#F3F4F6",
            highlightthickness=0,
            length=100,
        ).pack(side=tk.LEFT, padx=(2, 8))

        tk.Checkbutton(
            controls,
            text="Auto-recognize",
            variable=self._auto_recognize,
            bg="#0D0D10",
            fg="#F3F4F6",
            selectcolor="#26262B",
            activebackground="#0D0D10",
            font=("Helvetica", 10),
        ).pack(side=tk.LEFT)

        tk.Button(
            controls,
            text="Clear",
            command=self.clear,
            bg="#EF4444",
            fg="#FFFFFF",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=(4, 0))

        tk.Button(
            controls,
            text="Recognize",
            command=self.predict_canvas,
            bg="#10B981",
            fg="#FFFFFF",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            controls,
            text="Save",
            command=self.save_image,
            bg="#3B82F6",
            fg="#FFFFFF",
            relief=tk.FLAT,
            font=("Helvetica", 10, "bold"),
            cursor="hand2",
        ).pack(side=tk.RIGHT, padx=4)

        # Drawing canvas
        self._canvas = tk.Canvas(
            self,
            width=self.CANVAS_SIZE,
            height=self.CANVAS_SIZE,
            bg="white",
            cursor="crosshair",
            highlightthickness=2,
            highlightbackground="#26262B",
        )
        self._canvas.pack(padx=8, pady=8)

        # Bind drawing events
        self._canvas.bind("<B1-Motion>", self._on_draw)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Button-1>", self._on_press)
        self._canvas.bind("<B3-Motion>", self._on_erase)  # right-click = erase

    # Drawing event handlers

    def _on_press(self, event: tk.Event) -> None:
        self._prev_x = event.x
        self._prev_y = event.y

    def _on_draw(self, event: tk.Event) -> None:
        """Draw a thick oval stroke following the mouse."""
        r = self._brush_size.get() // 2
        x, y = event.x, event.y

        # Draw on Tkinter canvas
        self._canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="")

        # Mirror on PIL backing image (for export)
        self._pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=0)

        self._prev_x = x
        self._prev_y = y

    def _on_erase(self, event: tk.Event) -> None:
        """Erase (draw white) on right-click drag."""
        r = self._brush_size.get()
        x, y = event.x, event.y
        self._canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="")
        self._pil_draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    def _on_release(self, _: tk.Event) -> None:
        self._prev_x = None
        self._prev_y = None
        if self._auto_recognize.get():
            self.predict_canvas()

    # Public API

    def clear(self) -> None:
        """Clear the canvas and reset the PIL backing image."""
        self._canvas.delete("all")
        self._pil_image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 255)
        self._pil_draw = ImageDraw.Draw(self._pil_image)
        if self._on_clear:
            self._on_clear()

    def get_canvas_image(self) -> Image.Image:
        """Return the current canvas content as a 280×280 grayscale PIL image."""
        return self._pil_image.copy()

    def predict_canvas(self) -> None:
        """Capture canvas content and invoke the prediction callback."""
        if self._pil_image.getextrema() == (255, 255):
            if self._on_clear:
                self._on_clear()
            return

        if self._on_predict is not None:
            img = self.get_canvas_image()
            self._on_predict(img)

    def save_image(self) -> None:
        """Prompt to save the current canvas drawing to a file."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")],
            title="Save Image As"
        )
        if file_path:
            try:
                self.get_canvas_image().save(file_path)
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image:\n{e}")
