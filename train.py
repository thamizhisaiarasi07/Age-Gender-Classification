"""
train.py
=============================================================================
Age and Gender Classification from Facial Images Using Deep Learning
Enhanced Pipeline: Real UTKFace Dataset, Two-Stage Transfer Learning (MobileNetV2),
Class-Balanced Dataset, and Rigorous Test Set Evaluation.
=============================================================================
"""

import os
import sys
import glob
import json
import random
import tarfile
import urllib.request
import argparse
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# =============================================================================
# Configuration & Constants
# =============================================================================
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
STAGE1_EPOCHS = 8
STAGE2_EPOCHS = 5

# 7 Age Groups as specified
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
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "validation")
TEST_DIR = os.path.join(DATASET_DIR, "test")
ARCHIVE_PATH = os.path.join(DATASET_DIR, "UTKFace.tar.gz")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "age_gender_model.keras")
STATIC_DIR = os.path.join(BASE_DIR, "static")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
HISTORY_PLOT_PATH = os.path.join(STATIC_DIR, "training_history.png")

UTKFACE_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/main/UTKFace.tar.gz"
)


# =============================================================================
# Age & Filename Parsing
# =============================================================================
def age_to_group(age: int) -> int:
    """
    Maps an integer age into 7 discrete age brackets:
    0: 0-9 | 1: 10-19 | 2: 20-29 | 3: 30-39 | 4: 40-49 | 5: 50-59 | 6: 60+
    """
    if age <= 9:
        return 0
    elif age <= 19:
        return 1
    elif age <= 29:
        return 2
    elif age <= 39:
        return 3
    elif age <= 49:
        return 4
    elif age <= 59:
        return 5
    else:
        return 6


def parse_filename(filepath: str):
    """
    Parses age and gender from UTKFace filename convention:
    Format: [age]_[gender]_[race]_[date&time].jpg
    Example: '26_0_1_20170116171045112.jpg.chip.jpg' -> Age: 26 (Group: 2), Gender: 0 (Male)
    Returns: (age_group_idx, gender_idx, age_exact) or None if invalid.
    """
    filename = os.path.basename(filepath)
    parts = filename.split("_")
    if len(parts) >= 3:
        try:
            age = int(parts[0])
            gender = int(parts[1])
            if gender in (0, 1) and 0 <= age <= 120:
                age_group = age_to_group(age)
                return age_group, gender, age
        except ValueError:
            return None
    return None


# =============================================================================
# Automated Dataset Download & Balanced Preparation
# =============================================================================
def download_utkface_archive(archive_path: str):
    """Downloads the official UTKFace archive if not already available locally."""
    if os.path.exists(archive_path) and os.path.getsize(archive_path) > 10 * 1024 * 1024:
        print(f"[Info] Found existing UTKFace archive: {archive_path}")
        return True

    print("\n" + "="*70)
    print(" Downloading Real-World UTKFace Dataset (~101 MB)...")
    print(f" Source: {UTKFACE_DOWNLOAD_URL}")
    print("="*70)

    try:
        def _reporthook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\r Progress: {percent}% [{count * block_size // 1024 // 1024}MB / {total_size // 1024 // 1024}MB]")
            sys.stdout.flush()

        urllib.request.urlretrieve(UTKFACE_DOWNLOAD_URL, archive_path, reporthook=_reporthook)
        print("\n[SUCCESS] Download completed successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download UTKFace archive: {e}")
        print("Alternative: You can manually download UTKFace from Kaggle:")
        print("https://www.kaggle.com/datasets/jangedoo/utkface-new")
        print(f"and place the extracted files in: {DATASET_DIR}")
        return False


def prepare_balanced_dataset(archive_path: str, samples_per_gender_per_age: int = 300,
                             train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """
    Extracts, filters, and balances the UTKFace dataset:
    - 7 Age Groups: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+
    - Exactly equal number of Male and Female samples per age group.
    - Zero data leakage between Train, Validation, and Test splits.
    """
    print("\n" + "="*70)
    print(" Preparing Balanced UTKFace Dataset from Archive")
    print(f" Target: {samples_per_gender_per_age} Male + {samples_per_gender_per_age} Female per Age Group")
    print(f" Total target images: {samples_per_gender_per_age * 2 * len(AGE_GROUPS)}")
    print("="*70)

    pools = defaultdict(lambda: defaultdict(list))

    with tarfile.open(archive_path, "r:gz") as tar:
        members = tar.getmembers()
        print(f"Scanning {len(members)} entries in archive...")
        for m in members:
            if not m.isfile():
                continue
            base = os.path.basename(m.name)
            parsed = parse_filename(base)
            if parsed is not None:
                ag, gnd, _ = parsed
                pools[ag][gnd].append(m)

        # Check availability
        print("\nArchive image availability:")
        for ag in range(len(AGE_GROUPS)):
            print(f"  Age Group '{AGE_GROUPS[ag]}': Male={len(pools[ag][0])}, Female={len(pools[ag][1])}")

        # Clear existing images
        for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*.jpg")):
                try:
                    os.remove(f)
                except OSError:
                    pass

        # Split and extract
        total_extracted = 0
        for ag in range(len(AGE_GROUPS)):
            for gnd in [0, 1]:
                candidates = pools[ag][gnd]
                random.shuffle(candidates)
                selected = candidates[:samples_per_gender_per_age]

                n_train = int(len(selected) * train_ratio)
                n_val = int(len(selected) * val_ratio)
                train_items = selected[:n_train]
                val_items = selected[n_train:n_train + n_val]
                test_items = selected[n_train + n_val:]

                for item in train_items:
                    tar.extract(item, path=DATASET_DIR)
                    src = os.path.join(DATASET_DIR, item.name)
                    dst = os.path.join(TRAIN_DIR, os.path.basename(item.name))
                    if os.path.exists(src) and src != dst:
                        os.replace(src, dst)

                for item in val_items:
                    tar.extract(item, path=DATASET_DIR)
                    src = os.path.join(DATASET_DIR, item.name)
                    dst = os.path.join(VAL_DIR, os.path.basename(item.name))
                    if os.path.exists(src) and src != dst:
                        os.replace(src, dst)

                for item in test_items:
                    tar.extract(item, path=DATASET_DIR)
                    src = os.path.join(DATASET_DIR, item.name)
                    dst = os.path.join(TEST_DIR, os.path.basename(item.name))
                    if os.path.exists(src) and src != dst:
                        os.replace(src, dst)

                total_extracted += len(selected)

    # Clean any leftover extracted subfolders (e.g. dataset/UTKFace)
    subfolder = os.path.join(DATASET_DIR, "UTKFace")
    if os.path.exists(subfolder):
        import shutil
        shutil.rmtree(subfolder, ignore_errors=True)

    print(f"\n[SUCCESS] Successfully prepared {total_extracted} balanced real face images!")


# =============================================================================
# Dataset Statistics
# =============================================================================
def scan_dataset(directory: str):
    """Scans directory and returns filepaths, genders, and age group labels."""
    filepaths = []
    genders = []
    ages = []

    if not os.path.exists(directory):
        return filepaths, genders, ages

    valid_extensions = (".jpg", ".jpeg", ".png")
    for fname in os.listdir(directory):
        if fname.lower().endswith(valid_extensions):
            full_path = os.path.join(directory, fname)
            parsed = parse_filename(fname)
            if parsed is not None:
                ag_grp, gnd, _ = parsed
                filepaths.append(full_path)
                genders.append(gnd)
                ages.append(ag_grp)

    return filepaths, genders, ages


def print_dataset_statistics(train_g, train_a, val_g, val_a, test_g, test_a):
    """Prints comprehensive statistics and class balance breakdown."""
    print("\n" + "="*70)
    print("                     DATASET SPLIT STATISTICS")
    print("="*70)
    print(f"  Training Set:   {len(train_g)} images")
    print(f"  Validation Set: {len(val_g)} images")
    print(f"  Test Set:       {len(test_g)} images")
    print(f"  Total Images:   {len(train_g) + len(val_g) + len(test_g)} images")
    print("-" * 70)
    
    # Gender Distribution
    t_m, t_f = sum(1 for g in train_g if g == 0), sum(1 for g in train_g if g == 1)
    v_m, v_f = sum(1 for g in val_g if g == 0), sum(1 for g in val_g if g == 1)
    te_m, te_f = sum(1 for g in test_g if g == 0), sum(1 for g in test_g if g == 1)
    print("Gender Distribution:")
    print(f"  Train:      Male={t_m} ({t_m/len(train_g)*100:.1f}%), Female={t_f} ({t_f/len(train_g)*100:.1f}%)")
    print(f"  Validation: Male={v_m} ({v_m/len(val_g)*100:.1f}%), Female={v_f} ({v_f/len(val_g)*100:.1f}%)")
    print(f"  Test:       Male={te_m} ({te_m/len(test_g)*100:.1f}%), Female={te_f} ({te_f/len(test_g)*100:.1f}%)")
    print("-" * 70)

    # Age Group Distribution
    print("Age Group Distribution:")
    train_counts = Counter(train_a)
    val_counts = Counter(val_a)
    test_counts = Counter(test_a)
    print(f"  {'Age Group':<10} | {'Train':<8} | {'Validation':<10} | {'Test':<8} | {'Total':<8}")
    print("  " + "-" * 56)
    for idx, grp in enumerate(AGE_GROUPS):
        c_tr = train_counts.get(idx, 0)
        c_va = val_counts.get(idx, 0)
        c_te = test_counts.get(idx, 0)
        tot = c_tr + c_va + c_te
        print(f"  {grp:<10} | {c_tr:<8} | {c_va:<10} | {c_te:<8} | {tot:<8}")
    print("="*70 + "\n")


# =============================================================================
# TensorFlow Data Pipeline
# =============================================================================
def build_tf_dataset(filepaths, genders, ages, is_training: bool = True):
    """Builds optimized tf.data.Dataset with AUTOTUNE prefetching."""
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, genders, ages))

    def _process_item(path, gender, age):
        raw = tf.io.read_file(path)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        img.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
        return img, {"gender_output": gender, "age_output": age}

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 2000), seed=SEED)

    dataset = dataset.map(_process_item, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


# =============================================================================
# Model Architecture (Transfer Learning with MobileNetV2)
# =============================================================================
def build_age_gender_model(input_shape=(128, 128, 3)):
    """
    Constructs a dual-output architecture using MobileNetV2 pre-trained on ImageNet.
    Includes data augmentation, shared representation, and separate classification heads.
    """
    inputs = keras.Input(shape=input_shape, name="image_input")

    # 1. Realistic Data Augmentation (active only during training)
    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.06),
            layers.RandomZoom(0.06),
            layers.RandomTranslation(0.06, 0.06),
            layers.RandomBrightness(factor=0.08),
        ],
        name="data_augmentation"
    )
    x = data_augmentation(inputs)

    # 2. MobileNetV2 Normalization (scales pixels from [0, 255] to [-1, 1])
    x = keras.applications.mobilenet_v2.preprocess_input(x)

    # 3. Base MobileNetV2 Backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Initially frozen for Stage 1

    features = base_model(x, training=False)

    # 4. Shared Representation
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = layers.Dense(128, activation="relu", name="shared_dense")(x)
    x = layers.BatchNormalization(name="shared_batchnorm")(x)
    x = layers.Dropout(0.35, name="shared_dropout")(x)

    # 5. Gender Output Head (2 classes: Male, Female)
    gender_output = layers.Dense(
        len(GENDER_LABELS),
        activation="softmax",
        name="gender_output"
    )(x)

    # 6. Age Group Output Head (7 classes: 0-9 to 60+)
    age_output = layers.Dense(
        len(AGE_GROUPS),
        activation="softmax",
        name="age_output"
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=[gender_output, age_output],
        name="Age_Gender_MobileNetV2"
    )

    losses = {
        "gender_output": "sparse_categorical_crossentropy",
        "age_output": "sparse_categorical_crossentropy"
    }
    loss_weights = {"gender_output": 1.0, "age_output": 1.0}
    metrics = {
        "gender_output": ["accuracy"],
        "age_output": ["accuracy"]
    }

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )

    return model, base_model


# =============================================================================
# Two-Stage Training Function
# =============================================================================
def train_two_stages(model, base_model, train_ds, val_ds, stage1_epochs=STAGE1_EPOCHS, stage2_epochs=STAGE2_EPOCHS):
    """
    Executes a two-stage training strategy:
    Stage 1: Train classification heads with MobileNetV2 backbone frozen (lr=1e-3).
    Stage 2: Unfreeze upper layers of MobileNetV2 and fine-tune with low lr (1e-4).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    callbacks_stage1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=2, min_lr=1e-5, verbose=1)
    ]

    print("\n" + "="*70)
    print(f" STAGE 1: Training Classification Heads with Frozen Backbone ({stage1_epochs} Epochs)")
    print("="*70)
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1
    )

    print("\n" + "="*70)
    print(" STAGE 2: Fine-Tuning Top 30 MobileNetV2 Layers with Low LR (1e-4)")
    print("="*70)
    # Unfreeze top layers of MobileNetV2
    base_model.trainable = True
    fine_tune_at = len(base_model.layers) - 30
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    print(f"Total layers in base model: {len(base_model.layers)}")
    print(f"Trainable layers in Stage 2: {len(base_model.layers) - fine_tune_at}")

    # Recompile with small learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss={
            "gender_output": "sparse_categorical_crossentropy",
            "age_output": "sparse_categorical_crossentropy"
        },
        loss_weights={"gender_output": 1.0, "age_output": 1.0},
        metrics={
            "gender_output": ["accuracy"],
            "age_output": ["accuracy"]
        }
    )

    callbacks_stage2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2
    )

    # Combine histories
    combined_history = {}
    for k in history1.history.keys():
        combined_history[k] = history1.history[k] + history2.history[k]

    return combined_history


# =============================================================================
# Plotting & Evaluation
# =============================================================================
def plot_and_save_curves(history, save_path: str):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(epochs, history["loss"], label="Train Loss", marker="o", color="#2563eb")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", marker="s", color="#dc2626")
    axes[0].set_title("Total Model Loss", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Gender Acc
    axes[1].plot(epochs, history["gender_output_accuracy"], label="Train Gender Acc", marker="o", color="#059669")
    axes[1].plot(epochs, history["val_gender_output_accuracy"], label="Val Gender Acc", marker="s", color="#10b981")
    axes[1].set_title("Gender Classification Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # Age Acc
    axes[2].plot(epochs, history["age_output_accuracy"], label="Train Age Acc", marker="o", color="#7c3aed")
    axes[2].plot(epochs, history["val_age_output_accuracy"], label="Val Age Acc", marker="s", color="#a855f7")
    axes[2].set_title("Age Group Classification Accuracy", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Info] Training curves saved to: {save_path}")


def evaluate_on_test_set(test_paths, test_genders, test_ages):
    """
    Evaluates the trained model on unseen test data:
    - Generates confusion matrices
    - Calculates classification reports
    - Saves all plots to results/
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("\n" + "="*70)
    print("            EVALUATING MODEL ON UNSEEN TEST DATASET")
    print("="*70)

    model = keras.models.load_model(MODEL_PATH)
    test_ds = build_tf_dataset(test_paths, test_genders, test_ages, is_training=False)

    eval_results = model.evaluate(test_ds, verbose=1)
    # eval_results order: [loss, gender_loss, age_loss, gender_acc, age_acc]
    test_loss = eval_results[0]
    gender_acc = eval_results[3]
    age_acc = eval_results[4]

    # Predict test batch by batch
    all_gender_preds = []
    all_age_preds = []

    for batch_x, _ in test_ds:
        preds = model.predict(batch_x, verbose=0)
        g_pred = np.argmax(preds[0], axis=1)
        a_pred = np.argmax(preds[1], axis=1)
        all_gender_preds.extend(g_pred)
        all_age_preds.extend(a_pred)

    y_test_gender = np.array(test_genders)
    y_test_age = np.array(test_ages)
    y_pred_gender = np.array(all_gender_preds)
    y_pred_age = np.array(all_age_preds)

    # 1. Gender Classification Report & Matrix
    print("\n--- GENDER CLASSIFICATION REPORT ---")
    gender_report = classification_report(
        y_test_gender, y_pred_gender, target_names=GENDER_LABELS, digits=4
    )
    print(gender_report)

    gender_cm = confusion_matrix(y_test_gender, y_pred_gender)
    print("Gender Confusion Matrix:")
    print(gender_cm)

    # Plot Gender Confusion Matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(gender_cm, interpolation="nearest", cmap=plt.cm.Greens)
    plt.title("Gender Confusion Matrix", fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(GENDER_LABELS))
    plt.xticks(tick_marks, GENDER_LABELS)
    plt.yticks(tick_marks, GENDER_LABELS)
    for i in range(len(GENDER_LABELS)):
        for j in range(len(GENDER_LABELS)):
            plt.text(j, i, str(gender_cm[i, j]), horizontalalignment="center",
                     color="white" if gender_cm[i, j] > gender_cm.max() / 2 else "black", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    gender_cm_path = os.path.join(RESULTS_DIR, "gender_confusion_matrix.png")
    plt.savefig(gender_cm_path, dpi=200)
    plt.close()

    # 2. Age Group Classification Report & Matrix
    print("\n--- AGE GROUP CLASSIFICATION REPORT ---")
    age_report = classification_report(
        y_test_age, y_pred_age, target_names=AGE_GROUPS, digits=4, zero_division=0
    )
    print(age_report)

    age_cm = confusion_matrix(y_test_age, y_pred_age)
    print("Age Group Confusion Matrix:")
    print(age_cm)

    # Plot Age Confusion Matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(age_cm, interpolation="nearest", cmap=plt.cm.Purples)
    plt.title("Age Group Confusion Matrix", fontweight="bold")
    plt.colorbar()
    tick_marks = np.arange(len(AGE_GROUPS))
    plt.xticks(tick_marks, AGE_GROUPS, rotation=45)
    plt.yticks(tick_marks, AGE_GROUPS)
    for i in range(len(AGE_GROUPS)):
        for j in range(len(AGE_GROUPS)):
            plt.text(j, i, str(age_cm[i, j]), horizontalalignment="center",
                     color="white" if age_cm[i, j] > age_cm.max() / 2 else "black", fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    age_cm_path = os.path.join(RESULTS_DIR, "age_confusion_matrix.png")
    plt.savefig(age_cm_path, dpi=200)
    plt.close()

    # Save summary report text
    report_text_path = os.path.join(RESULTS_DIR, "evaluation_report.txt")
    with open(report_text_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write(" AGE & GENDER CLASSIFICATION: TEST EVALUATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Total Test Samples: {len(test_paths)}\n")
        f.write(f"Test Loss:          {test_loss:.4f}\n")
        f.write(f"Gender Accuracy:    {gender_acc*100:.2f}%\n")
        f.write(f"Age Group Accuracy: {age_acc*100:.2f}%\n\n")
        f.write("GENDER CLASSIFICATION REPORT:\n")
        f.write(gender_report + "\n\n")
        f.write("AGE GROUP CLASSIFICATION REPORT:\n")
        f.write(age_report + "\n")

    metrics_json_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump({
            "test_samples": len(test_paths),
            "test_loss": float(test_loss),
            "gender_accuracy": float(gender_acc),
            "age_accuracy": float(age_acc)
        }, f, indent=4)

    print("\n" + "="*70)
    print(f" [FINAL RESULTS] Gender Test Accuracy:    {gender_acc*100:.2f}%")
    print(f" [FINAL RESULTS] Age Group Test Accuracy: {age_acc*100:.2f}%")
    print(f" [FINAL RESULTS] Test Loss:               {test_loss:.4f}")
    print(f" [INFO] Detailed metrics saved in:        {RESULTS_DIR}")
    print("="*70 + "\n")

    return gender_acc, age_acc, test_loss


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train Age & Gender Classification Model")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download and prepare dataset without training")
    parser.add_argument("--samples-per-class", type=int, default=300,
                        help="Number of images per gender per age group (default: 300)")
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS,
                        help=f"Stage 1 epochs (default: {STAGE1_EPOCHS})")
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS,
                        help=f"Stage 2 fine-tuning epochs (default: {STAGE2_EPOCHS})")
    args = parser.parse_args()

    # 1. Dataset verification or automatic download & balance
    train_paths, train_g, train_a = scan_dataset(TRAIN_DIR)
    val_paths, val_g, val_a = scan_dataset(VAL_DIR)
    test_paths, test_g, test_a = scan_dataset(TEST_DIR)

    if len(train_paths) < 100 or len(test_paths) < 50:
        print("[!] Dataset splits incomplete. Checking UTKFace archive...")
        if not os.path.exists(ARCHIVE_PATH):
            success = download_utkface_archive(ARCHIVE_PATH)
            if not success:
                sys.exit(1)
        prepare_balanced_dataset(ARCHIVE_PATH, samples_per_gender_per_age=args.samples_per_class)
        # Rescan
        train_paths, train_g, train_a = scan_dataset(TRAIN_DIR)
        val_paths, val_g, val_a = scan_dataset(VAL_DIR)
        test_paths, test_g, test_a = scan_dataset(TEST_DIR)

    if args.download_only:
        print("[Info] --download-only completed. Dataset is ready.")
        return

    # 2. Print Dataset Statistics
    print_dataset_statistics(train_g, train_a, val_g, val_a, test_g, test_a)

    # 3. Create TensorFlow Datasets
    train_ds = build_tf_dataset(train_paths, train_g, train_a, is_training=True)
    val_ds = build_tf_dataset(val_paths, val_g, val_a, is_training=False)

    # 4. Build Model
    print("\n[Model] Building MobileNetV2 Dual-Head Architecture...")
    model, base_model = build_age_gender_model(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # 5. Two-Stage Training
    history = train_two_stages(
        model, base_model, train_ds, val_ds,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs
    )

    # 6. Save Plot
    plot_and_save_curves(history, HISTORY_PLOT_PATH)

    # 7. Evaluate on Unseen Test Set
    evaluate_on_test_set(test_paths, test_g, test_a)


if __name__ == "__main__":
    main()
