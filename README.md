# 🦴 Bone Fracture Classification with EfficientNet & Streamlit

A deep learning project to classify different types of bone fractures using X-ray images. This solution leverages **EfficientNetB0** for transfer learning and provides an easy-to-use **Streamlit UI** for live predictions.

---

## 📂 Dataset

- **Location**: `Bone Break Classification/`
- **Total Images**: 1129
- **Classes (10)**:
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

## 🧠 Model Architecture

- **Base**: EfficientNetB0 (pretrained on ImageNet)
- **Input Size**: 224x224x3
- **Layers Added**:
  - GlobalAveragePooling2D
  - Dropout
  - Dense (softmax)

### Training Setup

- Loss: `categorical_crossentropy`
- Optimizer: `Adam`
- Epochs: `10` + fine-tuning
- Class weights: Computed to address imbalance
- Data Augmentation: Flip, Rotate, Zoom

---

## 📈 Results

| Metric           | Value      |
|------------------|------------|
| Training Accuracy| ~44%       |
| Validation Accuracy | ~34%   |
| Model Confidence | Often <50% |

> 🔍 **Note**: Due to limited and imbalanced data, model predictions are not highly confident. Accuracy may improve with more data and tuning.

---

## 💻 Streamlit UI

### Features

- Upload X-ray images (`.jpg`, `.jpeg`, `.png`)
- Get predicted fracture type and confidence
- Alerts if confidence < 50%

### Run the App

```bash
streamlit run app.py
