"""
predict.py
=============================================================================
Age and Gender Classification - Standalone Image Inference Script
Enhanced to support:
- 7 Age Groups (0–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60+)
- MULTIPLE FACE DETECTION: Detects, crops, and predicts EACH face separately!
- Bounding box annotations and clear confidence reporting.
=============================================================================
"""

import os
import sys
import glob
import argparse
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# =============================================================================
# Constants & Labels
# =============================================================================
IMAGE_SIZE = (128, 128)

AGE_GROUPS = [
    "0–9",    # Index 0
    "10–19",  # Index 1
    "20–29",  # Index 2
    "30–39",  # Index 3
    "40–49",  # Index 4
    "50–59",  # Index 5
    "60+"     # Index 6
]

GENDER_LABELS = [
    "Male",    # Index 0
    "Female"   # Index 1
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "age_gender_model.keras")

# Distinct box colors for visualizing multiple faces
FACE_COLORS = [
    (0, 230, 0),     # Bright Green
    (255, 140, 0),   # Deep Sky Blue (BGR)
    (0, 165, 255),   # Orange
    (238, 130, 238), # Violet
    (255, 215, 0),   # Gold
]


# =============================================================================
# Face Detection & Inference
# =============================================================================
def load_face_cascade():
    """Loads OpenCV's pre-trained Haar Cascade classifier."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path):
        raise FileNotFoundError(f"Haar cascade XML not found at: {cascade_path}")
    return cv2.CascadeClassifier(cascade_path)


def predict_image(image_path: str, model_path: str = DEFAULT_MODEL_PATH, save_output: bool = True, force: bool = False):
    """
    Performs multi-face detection and separate age/gender inference:
    1. Loads the trained Keras model.
    2. Reads image with OpenCV.
    3. Detects all human faces using Haar Cascade.
    4. For EACH detected face:
       - Crops and preprocesses (128x128).
       - Predicts gender and age group with confidence percentages.
       - Annotates with bounding box and label.
    5. Saves annotated image and prints clear summary.
    """
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found at: {model_path}")
        print("Please train the model first by running: python train.py")
        return None

    if not os.path.exists(image_path):
        print(f"[ERROR] Image file does not exist: {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to read image (corrupted or unsupported format): {image_path}")
        return None

    orig_h, orig_w = image.shape[:2]

    # Face Detection with OpenCV Haar Cascade
    face_cascade = load_face_cascade()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(28, 28)
    )

    if len(faces) == 0:
        if not force:
            print("\n" + "="*60)
            print("[!] No human face detected by Haar Cascade in the image.")
            print("    Tips: Ensure the face is well-lit, clearly visible,")
            print("    and facing towards the camera.")
            print("    (If the image is already a close-up face crop, run with --force)")
            print("="*60)
            return {
                "face_detected": False,
                "message": "No face detected in the image.",
                "predictions": []
            }
        else:
            print("[Info] No Haar face box found, but --force passed. Analyzing whole image as face.")
            faces = np.array([[0, 0, orig_w, orig_h]])

    print(f"\n[Model] Loading trained model from: {model_path}...")
    model = keras.models.load_model(model_path)

    # Sort faces by x-coordinate (left to right) for clean enumeration
    faces = sorted(faces, key=lambda b: b[0])

    print("\n" + "="*65)
    print(f"       PREDICTION RESULTS (Detected {len(faces)} Face{'s' if len(faces)>1 else ''})")
    print("="*65)

    annotated = image.copy()
    predictions_list = []

    for i, (x, y, w, h) in enumerate(faces):
        # 12% padding for facial context
        pad_x = int(0.12 * w)
        pad_y = int(0.12 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(orig_w, x + w + pad_x)
        y2 = min(orig_h, y + h + pad_y)

        face_crop = image[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMAGE_SIZE)
        input_tensor = np.expand_dims(face_resized.astype(np.float32), axis=0)

        preds = model.predict(input_tensor, verbose=0)
        gender_preds = preds[0][0]
        age_preds = preds[1][0]

        # Extract predictions
        g_idx = int(np.argmax(gender_preds))
        g_conf = float(gender_preds[g_idx]) * 100.0
        g_label = GENDER_LABELS[g_idx]

        a_idx = int(np.argmax(age_preds))
        a_conf = float(age_preds[a_idx]) * 100.0
        a_label = AGE_GROUPS[a_idx]

        face_num = i + 1
        print(f"Face #{face_num}:")
        print(f"  Gender:     {g_label} ({g_conf:.1f}% confidence)")
        print(f"  Age Group:  {a_label} ({a_conf:.1f}% confidence)")
        print(f"  Coordinates: [x={x}, y={y}, w={w}, h={h}]")
        print("-" * 65)

        # Draw Bounding Box & Label
        color = FACE_COLORS[i % len(FACE_COLORS)]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        label_text = f"#{face_num}: {g_label} ({g_conf:.0f}%) | {a_label} ({a_conf:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        banner_y1 = max(0, y - th - 8)
        banner_y2 = y
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 8, banner_y2), (0, 0, 0), -1)
        cv2.putText(annotated, label_text, (x + 4, y - 4), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        predictions_list.append({
            "face_id": face_num,
            "gender": g_label,
            "gender_confidence": round(g_conf, 1),
            "age_group": a_label,
            "age_confidence": round(a_conf, 1),
            "box": (int(x), int(y), int(w), int(h))
        })

    print("="*65 + "\n")

    output_path = None
    if save_output:
        base_name, _ = os.path.splitext(image_path)
        output_path = f"{base_name}_predicted.jpg"
        cv2.imwrite(output_path, annotated)
        print(f"[Info] Annotated image saved to: {output_path}")

    return {
        "face_detected": True,
        "count": len(faces),
        "predictions": predictions_list,
        "output_path": output_path
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Age and Gender from an Image")
    parser.add_argument("--image", "-i", type=str, required=False,
                        help="Path to the input facial image")
    parser.add_argument("--model", "-m", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to the saved .keras model file")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Analyze whole image as face crop if no face box detected")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        # Check test images
        sample_candidates = glob.glob(os.path.join(BASE_DIR, "dataset", "test", "*.jpg"))
        if not sample_candidates:
            sample_candidates = glob.glob(os.path.join(BASE_DIR, "dataset", "train", "*.jpg"))
        if sample_candidates:
            image_path = sample_candidates[0]
            print(f"[Notice] No image provided via --image. Using test sample: {image_path}")
        else:
            print("[ERROR] Please provide an image path: python predict.py --image path/to/image.jpg")
            sys.exit(1)

    predict_image(image_path=image_path, model_path=args.model, force=args.force)


if __name__ == "__main__":
    main()
