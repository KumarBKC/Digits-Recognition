# Handwritten Digit Recognition

A PyTorch-based CNN for recognizing handwritten digits (0–9), with an interactive desktop UI for real-time inference via drawing, webcam, and image upload.

## Features

- **Drawing canvas** — draw digits directly and get instant predictions.
- **Webcam input** — recognize digits from a live camera feed.
- **Image upload** — load and classify digit images from disk.
- **Multi-digit recognition** — detects and classifies sequences of digits.
- **Confidence filtering** — predictions below 80% confidence are discarded.
- **Data augmentation** — expanded ~1,000 samples to ~52,000 using ±25° rotations.

## Installation

Requires **Python 3.8+**.

```bash
pip install -r requirements.txt
```

## Usage

Launch the desktop app:

```bash
python -m ui.main_app
```

**Keyboard Shortcuts:**

| Shortcut | Action |
|---|---|
| `Ctrl+S` / `Enter` | Run prediction |
| `Ctrl+Z` | Undo last stroke |
| `Ctrl+O` | Upload image |
| `Delete` / `Backspace` | Clear input |
| `Ctrl+Q` | Quit |

## Training

1. Place source images in `data/raw/` or `data/augmented/`.
2. Generate train/validation splits:
   ```bash
   python prepare_dataset.py
   ```
3. Augment the dataset (optional):
   ```bash
   python augment_data.py
   ```
4. Train the model:
   ```bash
   python train.py
   ```
5. Evaluate performance:
   ```bash
   python evaluate.py
   ```

## Project Structure

```
Digits_Recognition/
├── data/                   # Raw and processed datasets
├── images/                 # Training curves, confusion matrix
├── inference/              # Prediction and preprocessing logic
│   ├── predictor.py
│   ├── preprocessor.py
│   └── webcam_stream.py
├── models/                 # CNN architecture and saved checkpoints
│   ├── cnn_model.py
│   └── checkpoints/
├── training/               # Training pipeline
│   ├── augmentation.py
│   ├── dataset_loader.py
│   ├── metrics.py
│   └── trainer.py
├── ui/                     # Desktop interface (Tkinter)
│   ├── main_app.py
│   ├── canvas_panel.py
│   ├── result_display.py
│   ├── upload_panel.py
│   └── webcam_panel.py
├── utils/                  # Logging and visualization helpers
│   ├── logger.py
│   └── visualizer.py
├── augment_data.py         # Standalone augmentation script
├── evaluate.py             # Model evaluation script
├── prepare_dataset.py      # Dataset split script
├── train.py                # Training entry point
└── requirements.txt
```

## Results

### Training Curves

![Training Curves](images/training.png)

*The model reaches over 99% training accuracy and nearly 100% validation accuracy with no signs of overfitting.*

### Confusion Matrix

![Confusion Matrix](images/confusion.png)

*Near-perfect classification across all 10 digits with minimal misclassifications.*