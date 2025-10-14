import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K

# --- Streamlit page config ---
st.set_page_config(page_title="Chest X-ray Classifier", page_icon="🫁", layout="centered")

ACCENT = "#EAB308"
TOP_K = 4

st.markdown(
    """
    <style>
    .stFileUploader { background-color: transparent !important; }
    div[data-baseweb="file-uploader"] { background-color: transparent !important; border: none !important; box-shadow: none !important; }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1><span>🩺Chest X-ray Disease Classifier</span></h1>", unsafe_allow_html=True)
st.caption("Educational demo only — not for medical diagnosis.")

with st.expander("What does this app do?", expanded=False):
    st.write(
        "Upload a chest X-ray. The app loads your trained DenseNet model, "
        "preprocesses the image, predicts the disease class, "
        "and shows confidence scores."
    )

# --- Model setup ---
MODEL_PATH = "best_densenet_4class.h5"
CLASS_LABELS = ['Pneumonia-Bacterial', 'Covid-19', 'Normal', 'Tuberculosis']
IMG_SIZE = (224, 224)
ACCEPT = ["jpg", "jpeg", "png"]

# Optional: focal loss (if your model used it)
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1 - y_pred, gamma)
        return K.sum(weight * ce, axis=-1)
    return loss

# --- Load DenseNet model ---
@st.cache_resource
def load_model_func():
    return keras.models.load_model(MODEL_PATH, custom_objects={"loss": focal_loss()})

try:
    model = load_model_func()
except Exception as e:
    st.error(f"❌ Could not load model.h5. Error: {e}")
    st.stop()

# --- Preprocessing ---
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

# --- Dynamic Precautions based on class + confidence ---
def get_precautions(predicted_class: str, confidence: float):
    base_precautions = {
        "Pneumonia-Bacterial": [
            "Get plenty of rest and fluids.",
            "Avoid smoking or exposure to smoke.",
            "Cover your mouth when coughing or sneezing."
        ],
        "Covid-19": [
            "Wear a mask in crowded places.",
            "Wash your hands frequently with soap and water.",
            "Stay isolated if you have symptoms."
        ],
        "Normal": [
            "Maintain a healthy lifestyle with exercise.",
            "Avoid smoking and pollution exposure.",
            "Eat a balanced diet for lung health."
        ],
        "Tuberculosis": [
            "Cover your mouth when coughing or sneezing.",
            "Avoid close contact with others until cleared by a doctor.",
            "Keep windows open for ventilation."
        ]
    }

    strong_precautions = {
        "Pneumonia-Bacterial": [
            "Consult a healthcare provider immediately if symptoms worsen.",
            "Monitor temperature and breathing closely."
        ],
        "Covid-19": [
            "Seek medical attention immediately for difficulty breathing or high fever.",
            "Stay isolated and inform close contacts."
        ],
        "Normal": [
            "Continue regular check-ups to maintain healthy lungs."
        ],
        "Tuberculosis": [
            "Follow up regularly with a healthcare provider.",
            "Take medications as prescribed and complete the course."
        ]
    }

    # Determine precautions based on confidence
    if confidence > 0.9:
        items = base_precautions.get(predicted_class, []) + strong_precautions.get(predicted_class, [])
    elif confidence > 0.7:
        items = base_precautions.get(predicted_class, [])
        items.append("consulting a doctor is recommended.")
    else:
        items = ["retesting or consultation is recommended."]

    if not items:
        items = ["No specific precautions available."]
    
    return "\n".join([f"- {i}" for i in items])

# --- Streamlit UI ---
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

# --- Prediction Display ---
c1, c2 = st.columns([2, 1])
with c1:
    st.success(f"Prediction: {pred_label}")
with c2:
    st.metric(label="Confidence", value=f"{confidence*100:.2f}%")

# --- Urgency Alert ---
if pred_label != "Normal" and confidence > 0.9:
    st.error("🚨 **URGENT:** High confidence in abnormal X-ray. Please seek immediate medical attention.")
elif pred_label != "Normal" and confidence > 0.7:
    st.warning("⚠️ **Notice:** Possible abnormality detected. Please consult a doctor for confirmation.")
else:
    st.info("✅ No urgent signs detected.")

# --- Precautions Section ---
st.markdown("### 🩹 Precautions & Health Advice")
precautions_text = get_precautions(pred_label, confidence)
st.success(precautions_text)

# --- Confidence bars ---
st.markdown("#### Confidence (Top-K)")
pairs = sorted(list(zip(CLASS_LABELS, probs)), key=lambda t: t[1], reverse=True)[:TOP_K]
bar_df = pd.DataFrame({
    "Label": [p[0] for p in pairs],
    "Confidence %": [float(p[1])*100 for p in pairs]
}).set_index("Label")
st.bar_chart(bar_df)

# --- Expanded view for all classes ---
with st.expander("See all class probabilities", expanded=False):
    full_df = pd.DataFrame({
        "Label": CLASS_LABELS,
        "Confidence %": [float(p)*100 for p in probs]
    }).sort_values("Confidence %", ascending=False).reset_index(drop=True)
    st.dataframe(full_df, use_container_width=True)

st.caption("This tool is not a medical device and is provided for educational purposes only.")
st.markdown('</div>', unsafe_allow_html=True)



