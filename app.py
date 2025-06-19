import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
MODEL_PATH = "fracture_classifier_finetuned.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Avulsion fracture', 'Comminuted fracture', 'Fracture Dislocation',
               'Greenstick fracture', 'Hairline Fracture', 'Impacted fracture',
               'Longitudinal fracture', 'Oblique fracture', 'Pathological fracture',
               'Spiral Fracture']

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# Streamlit UI
st.title("🦴 Bone Fracture Classifier")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    pred_idx = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    st.markdown(f"### 🧠 Predicted: **{CLASS_NAMES[pred_idx]}** ({confidence:.2f}% confidence)")

    if confidence < 50:
        st.warning("⚠️ The model is not confident in this prediction. Please try another image.")
