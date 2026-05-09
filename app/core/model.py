"""ML model loader and inference — EfficientNet for DR grading."""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import timm

from app.core.config import settings

# Add Image_Processing to path
sys.path.insert(0, settings.IMAGE_PROCESSING_PATH)

_model = None
_device = None


def load_model():
    """Load the pretrained EfficientNet model from disk."""
    global _model, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {_device}")

    # Load weights
    model_path = os.path.abspath(settings.MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"[WARN] Model file not found at: {model_path}")
        return

    checkpoint = torch.load(model_path, map_location=_device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Detect number of output classes from the saved classifier weights
    num_classes = 5
    if "classifier.4.weight" in state_dict:
        num_classes = state_dict["classifier.4.weight"].shape[0]
    elif "classifier.1.weight" in state_dict:
        num_classes = state_dict["classifier.1.weight"].shape[0]

    # Build timm EfficientNet-B4 — key names match those in the checkpoint
    # (conv_stem, bn1, blocks.*, conv_head, bn2, classifier.*)
    model = timm.create_model(
        "efficientnet_b4",
        pretrained=False,
        num_classes=0,  # remove default head; we'll load the custom one
    )

    # Reconstruct the custom 2-layer classifier saved in the checkpoint
    # Expected: Dropout → Linear(1792→512) → ReLU → Dropout → Linear(512→num_classes)
    num_features = model.num_features  # 1792 for B4
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.4),
        nn.Linear(512, num_classes),
    )

    model.load_state_dict(state_dict)
    model.eval()
    model.to(_device)

    _model = model
    print(f"[SUCCESS] Model loaded from {model_path} ({num_classes} classes)")


def predict(image_array: np.ndarray) -> dict:
    """
    Run inference on a preprocessed image array.
    
    Args:
        image_array: numpy array of shape (380, 380, 3), already preprocessed
                     by image_pipeline.preprocess_retinal_image()
    
    Returns:
        dict with grade (int), label (str), confidence (float), description (str)
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    # Convert HWC numpy → CHW tensor
    tensor = torch.from_numpy(image_array.transpose(2, 0, 1)).float().unsqueeze(0)
    tensor = tensor.to(_device)

    with torch.no_grad():
        outputs = _model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    grade = predicted.item()
    conf = confidence.item()
    all_probs = probabilities[0].cpu().numpy().tolist()

    return {
        "grade": grade,
        "label": settings.DR_GRADES[grade],
        "confidence": round(conf * 100, 2),
        "description": settings.DR_DESCRIPTIONS[grade],
        "all_probabilities": {
            settings.DR_GRADES[i]: round(all_probs[i] * 100, 2)
            for i in range(5)
        },
    }
