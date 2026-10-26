"""
app.py
=============================================================================
Flask Web Application: Age and Gender Classification from Facial Images
Supports:
- 7 Age Groups (0–9, 10–19, 20–29, 30–39, 40–49, 50–59, 60+)
- MULTI-FACE DETECTION: Predicts and displays EACH face separately!
- Bounding-box visualization and confidence breakdowns.
=============================================================================
"""

import os
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "age_gender_model.keras")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
IMAGE_SIZE = (128, 128)

AGE_GROUPS = [
    "0–9",
    "10–19",
    "20–29",
    "30–39",
    "40–49",
    "50–59",
    "60–69",
    "70+"
]
GENDER_LABELS = [
    "Male",    # Index 0
    "Female"   # Index 1
]

FACE_COLORS = [
    (0, 230, 0),     # Bright Green
    (255, 140, 0),   # Deep Sky Blue
    (0, 165, 255),   # Orange
    (238, 130, 238), # Violet
    (255, 215, 0),   # Gold
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "age-gender-classifier-secret-key-2026"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

_cached_model = None
_cached_cascade = None


def get_face_cascade():
    global _cached_cascade
    if _cached_cascade is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade XML missing: {cascade_path}")
        _cached_cascade = cv2.CascadeClassifier(cascade_path)
    return _cached_cascade


def get_model():
    global _cached_model
    if _cached_model is None:
        if not os.path.exists(MODEL_PATH):
            return None
        print(f"[Flask] Loading trained model from: {MODEL_PATH}")
        _cached_model = keras.models.load_model(MODEL_PATH)
    return _cached_model


def is_allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def process_and_predict(filepath: str):
    model = get_model()
    if model is None:
        return {
            "error": "Model file not found! Please train the model first by running 'python train.py'."
        }

    image = cv2.imread(filepath)
    if image is None:
        return {
            "error": "The uploaded file could not be decoded as an image. It may be corrupted."
        }

    h_orig, w_orig = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = get_face_cascade()

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(28, 28)
    )

    filename_base = os.path.splitext(os.path.basename(filepath))[0]

    if len(faces) == 0:
        return {
            "face_detected": False,
            "message": "No human face was detected in this image. Please upload a clear, front-facing portrait photo.",
            "original_image_url": f"uploads/{os.path.basename(filepath)}"
        }

    # Sort faces left to right
    faces = sorted(faces, key=lambda b: b[0])
    annotated = image.copy()
    face_results = []

    for i, (x, y, w, h) in enumerate(faces):
        pad_x = int(0.12 * w)
        pad_y = int(0.12 * h)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w_orig, x + w + pad_x)
        y2 = min(h_orig, y + h + pad_y)

        face_crop = image[y1:y2, x1:x2]

        # Save cropped face chip
        crop_filename = f"crop_{i+1}_{filename_base}.jpg"
        crop_path = os.path.join(UPLOAD_FOLDER, crop_filename)
        cv2.imwrite(crop_path, face_crop)

        # Preprocess
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, IMAGE_SIZE)
        input_tensor = np.expand_dims(face_resized.astype(np.float32), axis=0)

        preds = model.predict(input_tensor, verbose=0)
        gender_preds = preds[0][0]
        age_preds = preds[1][0]

        best_g_idx = int(np.argmax(gender_preds))
        gender_conf = float(gender_preds[best_g_idx]) * 100.0
        gender_label = GENDER_LABELS[best_g_idx]

        best_a_idx = int(np.argmax(age_preds))
        age_conf = float(age_preds[best_a_idx]) * 100.0
        age_label = AGE_GROUPS[best_a_idx]

        color = FACE_COLORS[i % len(FACE_COLORS)]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)

        label_text = f"#{i+1}: {gender_label} ({gender_conf:.0f}%) | {age_label} ({age_conf:.0f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
        banner_y1 = max(0, y - th - 10)
        banner_y2 = y
        cv2.rectangle(annotated, (x, banner_y1), (x + tw + 10, banner_y2), (0, 0, 0), -1)
        cv2.putText(annotated, label_text, (x + 5, y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        gender_breakdown = [
            {"label": GENDER_LABELS[j], "prob": round(float(gender_preds[j]) * 100.0, 1)}
            for j in range(len(GENDER_LABELS))
        ]

        age_breakdown = [
            {"label": AGE_GROUPS[j], "prob": round(float(age_preds[j]) * 100.0, 1)}
            for j in range(len(AGE_GROUPS))
        ]

        face_results.append({
            "face_id": i + 1,
            "gender": gender_label,
            "gender_confidence": round(gender_conf, 1),
            "age_group": age_label,
            "age_confidence": round(age_conf, 1),
            "crop_image_url": f"uploads/{crop_filename}",
            "gender_breakdown": gender_breakdown,
            "age_breakdown": age_breakdown
        })

    annotated_filename = f"annotated_{filename_base}.jpg"
    annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
    cv2.imwrite(annotated_path, annotated)

    return {
        "face_detected": True,
        "face_count": len(faces),
        "faces": face_results,
        "original_image_url": f"uploads/{os.path.basename(filepath)}",
        "annotated_image_url": f"uploads/{annotated_filename}"
    }


@app.route("/", methods=["GET", "POST"])
def index():
    model_exists = os.path.exists(MODEL_PATH)

    if request.method == "POST":
        if "image" not in request.files:
            flash("No file part in the upload request.", "error")
            return redirect(request.url)

        file = request.files["image"]
        if file.filename == "":
            flash("Please choose an image file to upload.", "warning")
            return redirect(request.url)

        if not is_allowed_file(file.filename):
            flash("Unsupported file format. Please upload JPG, JPEG, or PNG images.", "error")
            return redirect(request.url)

        ext = file.filename.rsplit(".", 1)[1].lower()
        unique_name = f"{uuid.uuid4().hex[:10]}.{ext}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        file.save(filepath)

        result = process_and_predict(filepath)

        if "error" in result:
            flash(result["error"], "error")
            return render_template("index.html", model_exists=model_exists, result=None)

        return render_template("index.html", model_exists=model_exists, result=result)

    return render_template("index.html", model_exists=model_exists, result=None)


if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Age and Gender Classification Web Application")
    print(" Running at: http://127.0.0.1:5000/")
    print(f" Trained model status: {'AVAILABLE' if os.path.exists(MODEL_PATH) else 'NOT FOUND (Run train.py)'}")
    print("="*60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
