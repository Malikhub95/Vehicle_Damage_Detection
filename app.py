# app.py
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import tempfile
import torch


import streamlit as st

try:
    from ultralytics import YOLO
    import torch
    from PIL import Image
    import tempfile
    import os
except Exception as e:
    st.error(f"⚠️ Import failed: {e}")
    st.stop()

# -----------------------------
# Load model once
# -----------------------------
MODEL_PATH = 'yolov8_model/best.pt'  # Adjust this path if needed

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Vehicle Damage Detection")
st.markdown("Upload an image of a vehicle, and the model will predict damage locations.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect Damage"):
        with st.spinner("Running detection..."):
            # Save image to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                image.save(temp.name)
                results = model.predict(
                    source=temp.name,
                    conf=0.10,
                    save=False,
                    save_txt=False,
                    device=0 if torch.cuda.is_available() else 'cpu'
                )

            # Show result
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result", use_column_width=True)

