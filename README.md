# Age and Gender Classification from Facial Images Using Deep Learning

**Academic Project** | B.Tech Artificial Intelligence and Data Science (3rd Year)  
**Architecture:** Shared MobileNetV2 Feature Extractor with Multi-Output Heads  
**Web Framework:** Flask + OpenCV + Vanilla HTML/CSS  

---

## 1. Project Description

This project implements an end-to-end Deep Learning system capable of detecting a human face from an input image and simultaneously classifying:
1. **Gender** (Male or Female)
2. **Age Group** (0–5, 6–12, 13–19, 20–29, 30–39, 40–49, 50–59, 60+)

Using **Transfer Learning** with a pre-trained **MobileNetV2** backbone, the project achieves high efficiency with low computational latency, making it ideal for student laptops and embedded deployments without dedicated GPUs. The system is bundled with a web interface built with Flask and styled with responsive CSS.

---

## 2. Key Features

- **Automated Face Localization**: OpenCV Haar Cascade detects the human face and handles multiple faces by prioritizing the most prominent face.
- **Multi-Task Learning**: A single neural network simultaneously predicts two distinct outputs (Gender and Age Group) from shared convolutional feature maps.
- **MobileNetV2 Backbone**: Lightweight inverted residual bottleneck architecture pre-trained on ImageNet.
- **On-the-Fly Data Augmentation**: Keras sequential augmentation (random rotation, horizontal flip, zoom, translation) active during training to prevent overfitting.
- **Robust Error Handling**: Gracefully handles missing faces, unsupported file formats, and corrupted uploads.
- **Interactive Web Interface**: Live preview before upload, bounding-box display, cropped face chip view, confidence scores, and probability distribution bars.
- **Built-in Dataset Preparation & Synthetic Generator**: Built-in scripts to organize raw UTKFace images or generate synthetic face samples for instant offline demonstration.

---

## 3. Technologies Used

- **Programming Language**: Python 3.11+
- **Deep Learning Framework**: TensorFlow 2.15+ / Keras 3
- **Computer Vision**: OpenCV (`cv2`)
- **Data Manipulation & Plotting**: NumPy, Pandas, Matplotlib, Pillow
- **Web Backend**: Flask, Werkzeug
- **Frontend**: Semantic HTML5, Vanilla CSS3, minimal JavaScript (FileReader API)

---

## 4. Dataset Information

### Recommended Dataset: UTKFace
- **Source**: [Kaggle - UTKFace New (jangedoo/utkface-new)](https://www.kaggle.com/datasets/jangedoo/utkface-new?utm_source=chatgpt.com)
- **Image Count**: Over 20,000 face images in diverse poses, lighting, and ethnicities.
- **Filename Convention**:
  ```text
  [age]_[gender]_[race]_[date&time].jpg
  ```
  - `age`: Integer from 0 to 116.
  - `gender`: `0` for Male, `1` for Female.
  - `race`: `0` (White), `1` (Black), `2` (Asian), `3` (Indian), `4` (Others).
  - *Example*: `26_0_1_20170116171045112.jpg` (26-year-old Male).

### Age Group Discretization
To ensure high classification stability, ages are grouped into 8 classes:
- `0–5` (Index 0)
- `6–12` (Index 1)
- `13–19` (Index 2)
- `20–29` (Index 3)
- `30–39` (Index 4)
- `40–49` (Index 5)
- `50–59` (Index 6)
- `60+` (Index 7)

### Handling Resource Limits ("Remember the Limit")
Training on the complete 23,000+ images on a standard student laptop can lead to RAM exhaustion and take hours on CPU. The project includes a `--limit` flag (default **5,000 samples**) to ensure fast convergence within 15–20 minutes.

---

## 5. Model Architecture & Pipeline

```
                       [ Input Image: 128x128x3 ]
                                   │
                     [ Data Augmentation (Train only) ]
                     (RandomFlip, Rotation, Zoom, Shift)
                                   │
                      [ MobileNetV2 Feature Extractor ]
                         (ImageNet weights, Frozen)
                                   │
                     [ GlobalAveragePooling2D (1280) ]
                                   │
                     [ Dense(128, ReLU) + BatchNorm ]
                                   │
                           [ Dropout (0.3) ]
                                   ├──────────────────────────┐
                                   │                          │
                        [ Gender Output Head ]      [ Age Output Head ]
                            Dense(2, Softmax)          Dense(8, Softmax)
                                   │                          │
                           Loss: Sparse CCE           Loss: Sparse CCE
```

### Complete End-to-End Pipeline:
1. **User uploads image** via Web UI.
2. **OpenCV Haar Cascade** detects face coordinates `(x, y, w, h)`.
3. **Face Cropping & Padding** extracts the face region with 12% border padding.
4. **Resize & Normalization** scales crop to $128 \times 128 \times 3$ and normalizes to $[-1, 1]$.
5. **MobileNetV2 Forward Pass** generates shared feature embeddings.
6. **Classification Heads** compute softmax probabilities for gender and age group.
7. **Flask renders results**: Displays annotated bounding box, cropped face, and confidence percentages.

---

## 6. Project Structure

```text
Age-Gender-Classification/
│
├── dataset/
│   ├── train/                  # Training images ([age]_[gender]_*.jpg)
│   └── validation/             # Validation images ([age]_[gender]_*.jpg)
│
├── models/
│   └── age_gender_model.keras  # Saved Keras model
│
├── static/
│   ├── style.css               # Modern CSS stylesheet
│   ├── training_history.png    # Training & validation metric plots
│   └── uploads/                # Temporary uploaded & annotated images
│
├── templates/
│   └── index.html              # Responsive Flask web interface
│
├── train.py                    # Dataset preparation & model training pipeline
├── predict.py                  # Standalone CLI image inference script
├── app.py                      # Flask web application backend
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation & viva guide
```

---

## 7. Windows Setup & Installation

### Step 1: Open PowerShell or Command Prompt
Navigate to your project root folder:
```powershell
cd C:\Users\Chart\.gemini\antigravity\scratch\Age-Gender-Classification
```

### Step 2: Create a Virtual Environment (Recommended)
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **PowerShell**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Command Prompt (CMD)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```

### Step 4: Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 8. Dataset Preparation & Training

You have two options to train the model:

### Option A: Quick Demonstration (Synthetic Faces)
Generates 240 synthetic face samples and trains immediately:
```powershell
python train.py --generate-sample --epochs 5
```

### Option B: Real Training using Kaggle UTKFace
1. Download the UTKFace dataset from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new?utm_source=chatgpt.com).
2. Extract the downloaded archive to a directory (e.g., `C:\utkface_raw`).
3. Run the automated dataset preparation and training:
```powershell
python train.py --prepare "C:\utkface_raw" --limit 5000 --epochs 15
```

*Note:* The `--limit 5000` parameter keeps memory usage low and ensures the training completes quickly on a student laptop.

---

## 9. Testing & Inference

### Command-Line Prediction (`predict.py`)
Run inference on any image:
```powershell
python predict.py --image "path\to\photo.jpg"
```
If the image is already a close-up face crop without background:
```powershell
python predict.py --image "path\to\photo.jpg" --force
```

### Start the Flask Web Application (`app.py`)
```powershell
python app.py
```
Open your browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 10. Expected Web Application Output

When an image is submitted:
1. **Annotated Image Display**: The uploaded image is displayed with a green bounding box around the detected face and an overlaid label.
2. **Normalized Face Chip**: A $128 \times 128$ preview of the exact cropped face fed to the neural network.
3. **Gender Card**: e.g., `Female` (Confidence: 94.2%)
4. **Age Group Card**: e.g., `20–29` (Confidence: 87.5%)
5. **Confidence Distribution**: Progress bars showing the predicted probabilities across all 8 age groups and 2 genders.

---

## 11. Error Handling

- **No Image Selected**: Alerts user to select a valid file before submitting.
- **Unsupported Format**: Rejects non-image files and displays an alert (only `.jpg`, `.jpeg`, `.png` allowed).
- **Corrupted File**: Catches decoding errors and returns a helpful message.
- **No Face Detected**: OpenCV Haar Cascade detects 0 faces $\rightarrow$ Displays a warning explaining the face is not visible/angled, while showing the uploaded image.
- **Multiple Faces**: Detects multiple faces, outlines all of them, and performs prediction on the primary (largest area) face.
- **Missing Model**: Displays a warning banner advising the user to run `train.py`.

---

## 12. Limitations

1. **Extreme Poses**: Haar Cascades are optimized for frontal faces. Profiles or extreme tilts (>45°) may fail detection.
2. **Heavy Occlusions**: Sunglasses, thick face masks, or heavy shadows can reduce age and gender classification accuracy.
3. **Age Discretization**: Predicting an age group (e.g., 20–29) rather than an exact continuous age avoids ambiguous boundaries but sacrifices single-year granularity.

---

## 13. Future Enhancements

1. **Real-time Webcam Streaming**: Integrate OpenCV VideoCapture or WebRTC for live face prediction.
2. **Fine-Tuning**: Unfreeze the top layers of MobileNetV2 with a small learning rate ($10^{-5}$) for further accuracy gains.
3. **Emotion & Ethnicity Multi-tasking**: Add third and fourth heads for facial expression and ethnicity classification.

---

## 14. College Viva / Exam Q&A Guide

### Q1: Why use MobileNetV2 instead of training a CNN from scratch?
**Answer:** Training a CNN from scratch requires hundreds of thousands of images and days of compute to learn low-level visual filters (edges, textures). MobileNetV2 is pre-trained on ImageNet (1.4 million images). It uses depthwise separable convolutions and inverted residuals, offering high feature representation capability with minimal parameter count (~2.4M parameters).

### Q2: Why is this called Multi-Task Learning?
**Answer:** Instead of training two separate models (one for age, one for gender), a single backbone extracts shared facial features. The shared dense layer feeds two separate output heads. This halves computation time, saves memory, and allows features learned for gender to assist age prediction and vice versa.

### Q3: Why use `sparse_categorical_crossentropy` instead of `categorical_crossentropy`?
**Answer:** `categorical_crossentropy` requires one-hot encoded targets (e.g., `[0, 0, 1, 0, 0, 0, 0, 0]`). `sparse_categorical_crossentropy` directly accepts integer class indices (e.g., `2`), reducing memory overhead while computing the exact same mathematical loss.

### Q4: How does OpenCV Haar Cascade work?
**Answer:** Haar Cascades use rectangular features (differences in pixel intensity between adjacent regions) and the AdaBoost learning algorithm combined with an attentional cascade. Sub-windows without facial features are rejected in early stages, making it fast and lightweight for real-time CPU detection.
