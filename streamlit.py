import io
import json
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="centered")

# -------------------- FIXED THEME (no sidebar) --------------------
ACCENT = "#EAB308"   # fixed accent color (was selectable in sidebar)
TOP_K = 5            # fixed number of bars (was slider in sidebar)

# Inject CSS (gradient header, glass cards, accent color)
st.markdown(f"""
<style>
:root {{
  --accent: {ACCENT};
}}
/* Page padding */
.block-container {{
  padding-top: 2.0rem !important;
  padding-bottom: 3.0rem !important;
  max-width: 980px;
}}
h1 span {{
  background: linear-gradient(90deg, var(--accent), #0ea5e9);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
/* Glass card */
.card {{
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.15);
  box-shadow: 0 10px 30px rgba(0,0,0,0.15);
  border-radius: 16px;
  padding: 20px 22px;
}}
/* Dark mode friendliness */
[data-tested="stAppViewContainer"] {{
  background: radial-gradient(1200px circle at 10% 10%, rgba(99,102,241,0.08), transparent 40%),
              radial-gradient(1000px circle at 90% 30%, rgba(14,165,233,0.09), transparent 40%);
}}
/* Accent elements */
.stProgress > div > div > div {{
  background-color: var(--accent) !important;
}}
.st-emotion-cache-1r6slb0, .st-emotion-cache-16idsys, .st-emotion-cache-1wmy9hl {{
  color: var(--accent) !important;
}}
.st-emotion-cache-1y4p8pa a, .st-emotion-cache-1y4p8pa strong {{
  color: var(--accent) !important;
}}
/* Hide native footer */
footer {{ visibility: hidden; }}
.small {{ opacity: .85; font-size: .92rem; }}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1><span>Chest X-ray Disease Classifier</span></h1>", unsafe_allow_html=True)
st.caption("Educational demo only — not for medical diagnosis.")

with st.expander("What does this app do?", expanded=False):
    st.write(
        "Upload a chest X-ray. The app loads a trained TensorFlow/Keras model, "
        "preprocesses the image like training, predicts the most likely disease class, "
        "and shows confidences for all classes."
    )
# -------------------- MODEL / LABELS --------------------
MODEL_PATH = "model.h5"
CLASS_LABELS = ['Covid-19', 'Emphysema', 'Normal', 'Pneumonia-Bacterial']  # << update if needed
IMG_SIZE = (224, 224)
ACCEPT = ["jpg", "jpeg", "png"]

@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH)

# Try to load once; show a friendly error if missing
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Could not load model.h5. Error: {e}")
    st.stop()

# -------------------- HELPERS --------------------
def preprocess(pil_img: Image.Image):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0) / 255.0   # match training Rescaling(1/255)
    return arr

def predict_classes(pil_img: Image.Image):
    x = preprocess(pil_img)
    logits = model.predict(x, verbose=0)
    probs = logits[0]  # shape: [num_classes]
    idx = int(np.argmax(probs))
    return idx, probs

# -------------------- CARD: UPLOAD + RESULT --------------------
st.markdown('<div class="card">', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a chest X-ray (JPG/PNG)", type=ACCEPT)
if not uploaded_file:
    st.info("Drag & drop an X-ray above or click Browse files to start.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# preview image
pil = Image.open(uploaded_file)
st.image(pil, caption="Uploaded X-ray", use_container_width=True)

with st.spinner("🔍 Analyzing X-ray…"):
    pred_idx, probs = predict_classes(pil)
    pred_label = CLASS_LABELS[pred_idx]
    confidence = float(probs[pred_idx])
# Top result as a metric
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
    full_df = (pd.DataFrame({"Label": CLASS_LABELS,
                             "Confidence %": [float(p)*100 for p in probs]})
                 .sort_values("Confidence %", ascending=False)
                 .reset_index(drop=True))
    st.dataframe(full_df, use_container_width=True)

st.caption("This tool is not a medical device and is provided for educational purposes only.")
st.markdown('</div>', unsafe_allow_html=True)
