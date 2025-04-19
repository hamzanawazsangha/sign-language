import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import gdown
import os
import time
from typing import Tuple, Optional

# CONFIGURATION
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
MODEL_URL = "https://drive.google.com/uc?id=1gwGG918XTWKST--gTuMtKDxnFyPFzgXV"
MODEL_PATH = "final_sign.keras"
CLASS_NAMES = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
IMAGE_SIZE = (224, 224)
MIN_CONFIDENCE = 0.7

# ======================
# MODEL FUNCTIONS
# ======================
@st.cache_resource(show_spinner=False)
def load_model() -> Optional[tf.keras.Model]:
    """Download and load the model if not available locally."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            try:
                gdown.download(url=MODEL_URL, output=MODEL_PATH, quiet=False, use_cookies=False)
            except Exception as e:
                st.error(f"Model download failed: {str(e)}")
                return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize, normalize and add batch dimension."""
    image = image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image, model: tf.keras.Model) -> Tuple[str, float]:
    """Predict the sign from an image."""
    img_array = preprocess_image(image)
    start_time = time.time()
    preds = model.predict(img_array, verbose=0)
    inference_time = time.time() - start_time
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = CLASS_NAMES[class_idx]
    if confidence < MIN_CONFIDENCE:
        label = f"Low Confidence ({label})"
    st.sidebar.metric("Inference Time", f"{inference_time*1000:.1f}ms")
    st.sidebar.metric("Confidence Threshold", f"{MIN_CONFIDENCE*100:.0f}%")
    return label, confidence

# ======================
# STREAMLIT COMPONENTS
# ======================
def webcam_component(model: tf.keras.Model):
    st.info("Tips: Keep hand centered, ensure good lighting.")
    start = st.button("Start Webcam")
    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam.")
            return

        st.warning("Press 'Stop' in the sidebar to end webcam session.")
        stop = False

        while cap.isOpened():
            if st.sidebar.button("Stop Webcam"):
                stop = True
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Frame capture error.")
                break

            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            label, conf = predict(img_pil, model)
            cv2.putText(img_rgb, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            frame_placeholder.image(img_rgb, channels="RGB")

        cap.release()

def image_upload_component(model: tf.keras.Model):
    uploaded_file = st.file_uploader("Upload ASL Image", type=['jpg', 'jpeg', 'png', 'webp'])
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            with col2:
                brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
                enhancer = ImageEnhance.Brightness(img)
                img_adj = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(img_adj)
                img_adj = enhancer.enhance(contrast)
                st.image(img_adj, caption="Adjusted Image", use_column_width=True)

            with st.spinner("Analyzing..."):
                label, conf = predict(img_adj, model)
                confidence_color = "green" if conf >= MIN_CONFIDENCE else "orange"
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: #f8f9fa;">
                    <h3 style="color: {confidence_color};">Prediction: {label}</h3>
                    <div style="background: linear-gradient(90deg, #4CAF50 {conf*100}%, #e0e0e0 {conf*100}%); 
                                height: 25px; border-radius: 5px;"></div>
                    <p style="text-align: center; margin-top: 8px;">
                        Confidence: <strong>{conf*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# ======================
# MAIN APP
# ======================
def main():
    st.set_page_config(page_title="ASL Recognition Pro", page_icon="🤟", layout="wide")
    st.title("🤟 Advanced ASL Recognition System")
    st.markdown("Real-time American Sign Language detection. Upload images or use webcam!")

    model = load_model()
    if model is None:
        st.stop()

    tab1, tab2 = st.tabs(["🎥 Live Webcam Detection", "📷 Image Upload"])
    with tab1:
        webcam_component(model)
    with tab2:
        image_upload_component(model)

    st.markdown("---")
    st.caption("Note: This system recognizes standard ASL alphabets, numbers, and some special tokens.")

if __name__ == "__main__":
    main()
