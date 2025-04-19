import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown  # For downloading large files from Google Drive
import os

# Constants
MODEL_URL = "https://drive.google.com/uc?id=1UVHX3ePXl89Aeg6XxPg4QnyboGJ1SywJ"
MODEL_PATH = "final_sign.keras"
CLASS_NAMES = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
IMAGE_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load the model from local storage or download from Google Drive if not available."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive (this may take a while)..."):
            try:
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {str(e)}")
                return None
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess the image for model prediction."""
    img = image.resize(IMAGE_SIZE)
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image, model):
    """Make a prediction using the loaded model."""
    img_array = preprocess_image(image)
    preds = model.predict(img_array)
    class_idx = np.argmax(preds)
    confidence = np.max(preds)
    return CLASS_NAMES[class_idx], confidence

# Streamlit App Configuration
st.set_page_config(
    page_title="Sign Language Recognition System",
    page_icon="🤟",
    layout="centered"
)

# App Header
st.title("🤟 American Sign Language Recognition")
st.markdown("""
    This application detects American Sign Language (ASL) alphabets and numbers from images or webcam feed.
    Upload an image or use your webcam for real-time detection.
""")

# Load model with progress indicator
with st.spinner("Loading model..."):
    model = load_model()

if model is None:
    st.error("Failed to load model. Please check the model file and try again.")
    st.stop()

# Main Application Tabs
tab1, tab2 = st.tabs(["📸 Real-time Webcam Detection", "📂 Image Upload"])

with tab1:
    st.subheader("Real-time Sign Language Detection")
    st.info("""
        Click **Start Webcam** to begin real-time detection. 
        Ensure your hand signs are clearly visible in the frame.
    """)
    
    run = st.button("Start Webcam", key="start_webcam")
    stop = st.button("Stop Webcam", key="stop_webcam")
    FRAME_WINDOW = st.image([], caption="Live Detection Feed")

    if run and not stop:
        cap = cv2.VideoCapture(0)
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam. Please check your camera permissions.")
                break

            # Flip and convert frame
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Make prediction
            label, conf = predict(img_pil, model)
            
            # Display results
            cv2.putText(img_rgb, f'{label} ({conf:.2f})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            FRAME_WINDOW.image(img_rgb)

            if stop:
                cap.release()
                break

with tab2:
    st.subheader("Image Upload")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        accept_multiple_files=False,
        help="Upload a clear image of a hand sign for detection"
    )

    if uploaded_file is not None:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            with st.spinner("Analyzing sign..."):
                label, conf = predict(img, model)
                
            st.success(f"""
                **Prediction:** {label}  
                **Confidence:** {conf*100:.2f}%
            """)
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Footer
st.markdown("---")
st.caption("""
    Note: This application is designed for ASL alphabet and number recognition. 
    Performance may vary based on image quality and lighting conditions.
""")
