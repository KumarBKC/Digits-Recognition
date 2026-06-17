"""Export trained DigitCNN for optimized inference.

Supports:
  - INT8 dynamic quantization  (4x smaller, 2-3x faster on CPU)
  - ONNX export                (cross-platform, ONNX Runtime compatible)
  - TorchScript compilation    (C++ compatible, no Python dependency)
  - Inference benchmarking     (latency & throughput on CPU)

Usage:
    python export_model.py --all
    python export_model.py --quantize --benchmark
    python export_model.py --onnx --checkpoint models/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn

from models.cnn_model import DigitCNN, INPUT_HEIGHT, INPUT_WIDTH


def load_model(checkpoint_path: str, device: str = "cpu") -> DigitCNN:
    """Load a trained DigitCNN from a checkpoint file."""
    model = DigitCNN()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch:        {checkpoint.get('epoch', '?')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', float('nan')):.4f}")
    return model


def export_quantized(model: DigitCNN, output_path: str) -> None:
    """Dynamic INT8 quantization for CPU inference."""
    print("\n--- INT8 Dynamic Quantization ---")
    model_q = torch.quantization.quantize_dynamic(
        model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
    )
    torch.save(model_q.state_dict(), output_path)

    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    q_size = os.path.getsize(output_path)
    print(f"  Original model size: {orig_size / 1024:.1f} KB")
    print(f"  Quantized file size: {q_size / 1024:.1f} KB")
    print(f"  Compression ratio:   {orig_size / max(q_size, 1):.1f}x")
    print(f"  Saved to: {output_path}")


def export_onnx(model: DigitCNN, output_path: str) -> None:
    """Export to ONNX format for cross-platform inference."""
    print("\n--- ONNX Export ---")
    dummy_input = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
    torch.onnx.export(
        model,
        # pyrefly: ignore [bad-argument-type]
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    size = os.path.getsize(output_path)
    print(f"  ONNX file size: {size / 1024:.1f} KB")
    print(f"  Saved to: {output_path}")


def export_torchscript(model: DigitCNN, output_path: str) -> None:
    """Compile to TorchScript for C++ inference."""
    print("\n--- TorchScript Export ---")
    dummy_input = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)
    scripted = torch.jit.trace(model, dummy_input)
    scripted.save(output_path)
    size = os.path.getsize(output_path)
    print(f"  TorchScript file size: {size / 1024:.1f} KB")
    print(f"  Saved to: {output_path}")


def benchmark_inference(model: nn.Module, n_runs: int = 200) -> float:
    dummy = torch.randn(1, 1, INPUT_HEIGHT, INPUT_WIDTH)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            model(dummy)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / n_runs) * 1000
    return avg_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DigitCNN for deployment")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/checkpoints/best_model.pth",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/exported",
        help="Output directory for exported models.",
    )
    parser.add_argument(
        "--quantize", action="store_true", help="Export INT8 quantized model."
    )
    parser.add_argument("--onnx", action="store_true", help="Export ONNX model.")
    parser.add_argument(
        "--torchscript", action="store_true", help="Export TorchScript model."
    )
    parser.add_argument(
        "--benchmark", action="store_true", help="Benchmark inference speed."
    )
    parser.add_argument(
        "--all", action="store_true", help="Export all formats + benchmark."
    )
    args = parser.parse_args()

    if args.all:
        args.quantize = args.onnx = args.torchscript = args.benchmark = True

    if not any([args.quantize, args.onnx, args.torchscript, args.benchmark]):
        print(
            "No export format specified. "
            "Use --all, --quantize, --onnx, --torchscript, or --benchmark."
        )
        return

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint)

    if args.benchmark:
        print("\n--- Inference Benchmark (CPU) ---")
        avg_ms = benchmark_inference(model)
        print(f"  Average latency: {avg_ms:.2f} ms/image")
        print(f"  Throughput:      {1000 / avg_ms:.0f} images/sec")

    if args.quantize:
        export_quantized(
            model, os.path.join(args.output_dir, "digit_model_int8.pth")
        )

    if args.onnx:
        export_onnx(model, os.path.join(args.output_dir, "digit_model.onnx"))

    if args.torchscript:
        export_torchscript(
            model, os.path.join(args.output_dir, "digit_model_scripted.pt")
        )

    print("\n✓ Export complete!")


if __name__ == "__main__":
    main()
