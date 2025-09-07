import io
import json
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="centered")

ACCENT = "#EAB308"  
TOP_K = 5           

st.markdown(
    """
    <style>
    /* Remove the gray background of file uploader */
    .stFileUploader {
        background-color: transparent !important;
    }
    /* Remove the border around drag & drop area */
    div[data-baseweb="file-uploader"] {
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1><span>Chest X-ray Disease Classifier</span></h1>", unsafe_allow_html=True)
st.caption("Educational demo only — not for medical diagnosis.")

with st.expander("What does this app do?", expanded=False):
    st.write(
        "Upload a chest X-ray. The app loads a trained model, "
        "preprocesses the image, predicts the disease class, "
        "and shows confidence scores."
    )

MODEL_PATH = r"Model_Path"
CLASS_LABELS = ['Covid-19', 'Emphysema', 'Normal', 'Pneumonia-Bacterial']  
IMG_SIZE = (224, 224)
ACCEPT = ["jpg", "jpeg", "png"]

#focal loss
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * ce, axis=-1)
    return loss

@st.cache_resource
def load_model_func():
    return keras.models.load_model(MODEL_PATH, custom_objects={"loss": focal_loss()})

try:
    model = load_model_func()
except Exception as e:
    st.error(f"❌ Could not load model.h5. Error: {e}")
    st.stop()

def preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0
    return arr

def predict_classes(pil_img: Image.Image):
    x = preprocess(pil_img)
    logits = model.predict(x, verbose=0)
    probs = logits[0]
    idx = int(np.argmax(probs))
    return idx, probs

st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a chest X-ray (JPG/PNG)", type=ACCEPT)
if not uploaded_file:
    st.info("Drag & drop an X-ray above or click Browse files to start.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# Show uploaded image
pil = Image.open(uploaded_file)
st.image(pil, caption="Uploaded X-ray", use_container_width=True)

with st.spinner("🔍 Analyzing X-ray…"):
    pred_idx, probs = predict_classes(pil)
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(probs[pred_idx])

c1, c2 = st.columns([2, 1])
with c1:
    st.success(f"Prediction: {pred_label}")
with c2:
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

st.markdown("#### Confidence (Top-K)")
pairs = sorted(list(zip(CLASS_LABELS, probs)), key=lambda t: t[1], reverse=True)[:TOP_K]
bar_df = pd.DataFrame({"Label": [p[0] for p in pairs],
                       "Confidence %": [float(p[1])*100 for p in pairs]}).set_index("Label")
st.bar_chart(bar_df)

with st.expander("See all class probabilities", expanded=False):
    full_df = pd.DataFrame({"Label": CLASS_LABELS,
                            "Confidence %": [float(p)*100 for p in probs]}).sort_values("Confidence %", ascending=False).reset_index(drop=True)
    st.dataframe(full_df, use_container_width=True)

st.caption("This tool is not a medical device and is provided for educational purposes only.")
st.markdown('</div>', unsafe_allow_html=True)
