import cv2
import glob
import os

def score_sharpness(path):
    img = cv2.imread(path)
    if img is None:
        return 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Mild blur to suppress camera noise before Laplacian
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Laplacian(denoised, cv2.CV_64F).var()

def select_best_image(folder="images"):
    # Grab all jpg, jpeg, png — case-insensitive
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))

    if not paths:
        raise FileNotFoundError(f"No images found in: {folder}")

    scored = [(p, score_sharpness(p)) for p in paths]
    scored.sort(key=lambda x: x[1], reverse=True)

    print("Ranked images (sharpest first):")
    for p, s in scored:
        print(f"  {os.path.basename(p):30s}  score: {s:.2f}")

    best_path = scored[0][0]
    print(f"\nBest image: {best_path}")
    return best_path


if __name__ == "__main__":
    # ── Then feed directly into your existing pipeline ──
    best = select_best_image("images")
    # result = preprocess_retinal_image(best, target_size=380)