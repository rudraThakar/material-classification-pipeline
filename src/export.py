import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--class_index", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--out_torchscript", type=str, default="models/model.torchscript.pt")
    parser.add_argument("--out_onnx", type=str, default="models/model.onnx")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out_torchscript), exist_ok=True)

    with open(args.class_index, "r", encoding="utf-8") as f:
        class_index = json.load(f)
    num_classes = len(class_index)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state"])  # type: ignore[index]
    model.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)

    # TorchScript
    scripted = torch.jit.trace(model, dummy)
    scripted.save(args.out_torchscript)
    print(f"Saved TorchScript to {args.out_torchscript}")

    # ONNX
    torch.onnx.export(
        model,
        dummy,
        args.out_onnx,
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"Saved ONNX to {args.out_onnx}")


if __name__ == "__main__":
    main()


