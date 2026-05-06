import cv2
import numpy as np
from PIL import Image

# ─────────────────────────────────────────────────────────────
# STEP 1: Crop black border and center the retinal circle
# WHY: Vision capture will have large black areas around
#      the actual retina. This removes them automatically.
# ─────────────────────────────────────────────────────────────
def crop_retina(img, tolerance=7):
    """Auto-crop black borders around retinal image."""
    # Convert to grayscale to find the retinal circle
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find all pixels that are not black (threshold = 7)
    mask = gray > tolerance

    # Get bounding box of non-black region
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return img  # return original if fully black

    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    # Add 10px padding so edges aren't cut too tight
    h, w = img.shape[:2]
    y0, x0 = max(0, y0-10), max(0, x0-10)
    y1, x1 = min(h, y1+10), min(w, x1+10)

    return img[y0:y1, x0:x1]


# ─────────────────────────────────────────────────────────────
# STEP 2: Ben Graham preprocessing (winner of Kaggle 2015)
# WHY: Removes uneven illumination from the phone flash.
#      Phone flash is brighter in the center — this corrects it.
#      This is the EXACT method that won the original competition.
# ─────────────────────────────────────────────────────────────
def ben_graham_preprocess(img, sigmaX=10):
    """
    Ben Graham normalization:
    1. Create blurred version of image (captures uneven lighting)
    2. Subtract blur from original (removes the uneven lighting)
    3. Add back a grey base (so result isn't too dark)
    """
    # Gaussian blur captures the slow-varying illumination pattern
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX)

    # Subtract blur, add grey midpoint (128)
    result = cv2.addWeighted(img, 4, blur, -4, 128)

    # Apply circular mask — hide corners outside retinal circle
    h, w = result.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = w // 2, h // 2
    radius = min(cx, cy) - 10  # slight inset from edge
    cv2.circle(mask, (cx, cy), radius, 1, -1)

    # Apply mask — pixels outside circle become grey (128)
    result = result * mask[..., np.newaxis] + 128 * (1 - mask[..., np.newaxis])

    return result.astype(np.uint8)


# ─────────────────────────────────────────────────────────────
# STEP 3: CLAHE — Contrast Limited Adaptive Histogram Equalization
# WHY: Boosts LOCAL contrast so blood vessels and lesions
#      become visible even in darker images from Vision.
#      Applied to green channel only (best vessel contrast).
# ─────────────────────────────────────────────────────────────
def apply_clahe(img):
    """Apply CLAHE to the green channel for better vessel visibility."""
    # Split into BGR channels
    b, g, r = cv2.split(img)

    # CLAHE object: clipLimit controls how much contrast is boosted
    # tileGridSize: the size of local regions (8x8 is standard)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # Apply only to green channel (retinal vessels show best here)
    g_clahe = clahe.apply(g)

    # Recombine with original b and r channels
    return cv2.merge([b, g_clahe, r])


# ─────────────────────────────────────────────────────────────
# MASTER PIPELINE
# ─────────────────────────────────────────────────────────────
def preprocess_retinal_image(image_path, target_size=380):
    """
    Complete preprocessing pipeline.
    """
    # Load image
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
    else:
        img = image_path  # already numpy array

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Step 1: Crop black borders
    img = crop_retina(img)

    # Step 2: Resize to square target size FIRST
    #         (square resize, then circle mask in Ben Graham)
    img = cv2.resize(img, (target_size, target_size))

    # Step 3: Apply CLAHE on green channel
    img = apply_clahe(img)

    # Step 4: Ben Graham normalization (uneven light removal)
    img = ben_graham_preprocess(img, sigmaX=10)

    # Step 5: Convert BGR→RGB (OpenCV uses BGR, PyTorch uses RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 6: Normalize to ImageNet mean/std
    #         (EfficientNet was trained on these values — must match)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    return img  # shape: (380, 380, 3)