import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import onnxruntime as ort


def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_class_index(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    return {int(k): v for k, v in mapping.items()} if isinstance(mapping, dict) else mapping


def infer_torchscript(model_path: str, image_path: str, img_size: int) -> Tuple[int, float]:
    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()
    tf = build_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred = torch.max(probs, dim=0)
    return int(pred.item()), float(conf.item())


def infer_onnx(model_path: str, image_path: str, img_size: int) -> Tuple[int, float]:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])  # Jetson: use TensorRT if available
    tf = build_transform(img_size)
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).numpy()
    outputs = sess.run([sess.get_outputs()[0].name], {sess.get_inputs()[0].name: x})
    logits = outputs[0]
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pred = int(np.argmax(probs, axis=1)[0])
    conf = float(probs[0, pred])
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--class_index", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--low_conf_threshold", type=float, default=0.5)
    args = parser.parse_args()

    class_index = load_class_index(args.class_index)
    is_onnx = args.model.lower().endswith(".onnx")

    pred_idx, conf = (infer_onnx(args.model, args.image, args.img_size)
                      if is_onnx else infer_torchscript(args.model, args.image, args.img_size))
    label = class_index.get(pred_idx, str(pred_idx))
    low_flag = conf < args.low_conf_threshold
    print({"pred": label, "conf": round(conf, 4), "low_confidence": low_flag})


if __name__ == "__main__":
    main()


