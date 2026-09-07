"""Run a pretrained U-Net for human/person segmentation on one image."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = ROOT / "input" / "pic-555x740-6-1.jpg"
DEFAULT_CHECKPOINT = ROOT / "pretrained_unet_human.pth"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError("Checkpoint không chứa state_dict PyTorch hợp lệ.")

    state = dict(payload)
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_model(checkpoint: Path, device: torch.device) -> torch.nn.Module:
    # This is the same architecture family as the published checkpoint:
    # U-Net with a ResNet34 encoder and one binary output channel.
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = extract_state_dict(payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint không khớp kiến trúc U-Net ResNet34. "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    model.eval()
    return model


def preprocess(image: Image.Image, max_side: int) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Resize without distortion and pad to a multiple of the encoder stride."""

    original_width, original_height = image.size
    scale = max_side / max(original_width, original_height)
    resized_width = max(32, round(original_width * scale))
    resized_height = max(32, round(original_height * scale))
    padded_width = math.ceil(resized_width / 32) * 32
    padded_height = math.ceil(resized_height / 32) * 32
    left = (padded_width - resized_width) // 2
    top = (padded_height - resized_height) // 2

    resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
    letterbox = Image.new("RGB", (padded_width, padded_height), (0, 0, 0))
    letterbox.paste(resized, (left, top))
    array = np.asarray(letterbox, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    crop = (left, top, resized_width, resized_height)
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0), crop


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrained U-Net human segmentation")
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    parser.add_argument("--size", type=int, default=256, help="Longest side; aspect ratio is preserved")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {args.image}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Không tìm thấy pretrained checkpoint: {args.checkpoint}")

    device = pick_device(args.device)
    original = Image.open(args.image).convert("RGB")
    model = load_model(args.checkpoint, device)

    with torch.inference_mode():
        tensor, crop = preprocess(original, args.size)
        logits = model(tensor.to(device))
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    left, top, resized_width, resized_height = crop
    probability = probability[top : top + resized_height, left : left + resized_width]
    mask = Image.fromarray((probability >= args.threshold).astype(np.uint8) * 255)
    mask = mask.resize(original.size, Image.Resampling.NEAREST)
    args.output.mkdir(parents=True, exist_ok=True)

    mask_path = args.output / f"{args.image.stem}_mask.png"
    overlay_path = args.output / f"{args.image.stem}_overlay.png"
    mask.save(mask_path)

    red = Image.new("RGBA", original.size, (255, 40, 40, 0))
    red.putalpha(mask.point(lambda value: value * 120 // 255))
    Image.alpha_composite(original.convert("RGBA"), red).convert("RGB").save(overlay_path)

    print(f"Device: {device}")
    print(f"Foreground ratio: {float(np.asarray(mask).mean() / 255.0):.2%}")
    print(f"Mask: {mask_path}")
    print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
