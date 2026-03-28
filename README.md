# Handwritten Digit Recognition System

A complete end-to-end Handwritten Digit Recognition System built with a custom CNN in PyTorch.
Supports real-time webcam recognition, static image upload, and on-screen drawing from a single Tkinter desktop GUI.

## Features

- Custom CNN: Lightweight DigitCNN trained on 17x43 grayscale images
- 82% accuracy: Achieved on held-out validation set
- Live webcam: Bounding-box detection and classification in real time
- Image upload: JPG, PNG, BMP, TIFF support
- Drawing canvas: Draw with mouse and auto-recognised
- Desktop GUI: Tkinter application
- Checkpoint export: Auto saves best_model.pth during training

## Project Structure

```
digit_recognition/
+-- data/
¦   +-- dataset/
¦   +-- augmented/
¦   +-- mnist_supplement/
+-- models/
¦   +-- cnn_model.py
¦   +-- checkpoints/
¦       +-- best_model.pth
+-- training/
¦   +-- dataset_loader.py
¦   +-- augmentation.py
¦   +-- trainer.py
¦   +-- metrics.py
+-- inference/
¦   +-- predictor.py
¦   +-- webcam_stream.py
¦   +-- preprocessor.py
+-- ui/
¦   +-- main_app.py
¦   +-- canvas_panel.py
¦   +-- webcam_panel.py
¦   +-- upload_panel.py
¦   +-- result_display.py
+-- utils/
¦   +-- visualizer.py
¦   +-- logger.py
+-- train.py
+-- evaluate.py
+-- requirements.txt
```

## Dataset Structure

```
data/dataset/
+-- train/
¦   +-- 0/
¦   +-- 1/
¦   +-- 9/
+-- val/
    +-- 0/
    +-- 1/
    +-- 9/
```

Format: Grayscale PNG, 17x43 pixels

## CNN Architecture

```
Input: [B, 1, 43, 17]
Block 1: Conv2d(1->32) -> BN -> ReLU -> MaxPool(2)
Block 2: Conv2d(32->64) -> BN -> ReLU -> MaxPool(2)
Block 3: Conv2d(64->128) -> BN -> ReLU -> MaxPool(2)
Flatten: 1280
FC1: Linear(1280->256) -> ReLU -> Dropout
FC2: Linear(256->128) -> ReLU -> Dropout
FC3: Linear(128->10)
```

## Setup

```bash
pip install -r requirements.txt
python -m training.dataset_loader --validate --data_root ./data/dataset
```

## Training

```bash
python train.py --data_root ./data/dataset --epochs 60 --batch_size 32 --lr 0.001
```

## Evaluation

```bash
python evaluate.py --checkpoint models/checkpoints/best_model.pth --data_root ./data/dataset
```

## Running the UI

```bash
python ui/main_app.py --model models/checkpoints/best_model.pth
```
