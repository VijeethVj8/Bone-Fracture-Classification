# 🦴 Bone Fracture Classification using EfficientNet and Streamlit

This project aims to classify different types of bone fractures from X-ray images using a deep learning model built on top of **EfficientNet**. It also includes a **Streamlit-based UI** for testing predictions with real X-ray images.

---

## 📁 Dataset

- **Location**: `/Users/vijeethvj8/Downloads/Elevateme/Bone Break Classification`
- **Structure**: Folder contains subdirectories for 10 different fracture types:
  - Avulsion fracture
  - Comminuted fracture
  - Fracture Dislocation
  - Greenstick fracture
  - Hairline Fracture
  - Impacted fracture
  - Longitudinal fracture
  - Oblique fracture
  - Pathological fracture
  - Spiral Fracture

---

## 🧠 Model

- **Base Model**: `EfficientNetB0` (ImageNet pretrained)
- **Input Size**: `224x224`
- **Training Setup**:
  - Data Augmentation: Flip, Rotate, Zoom
  - Class Weights for imbalance handling
  - Categorical Crossentropy loss
  - Adam optimizer
  - 10–20 Epochs with fine-tuning
- **Fine-tuning**:
  - Top layers of EfficientNetB0 were unfrozen and trained with a lower learning rate
- **Performance**:
  - Best Accuracy Achieved: ~44% (Train), ~34% (Validation)
  - Class-wise imbalance exists

---

## 💻 App (Streamlit UI)

Users can upload X-ray images and receive predictions with confidence scores.

### Features:
- Upload `.jpg`, `.jpeg`, or `.png` images
- Displays predicted fracture type
- Confidence score shown
- Low-confidence warnings (<50%)

### Run the App:

```bash
streamlit run app.py
