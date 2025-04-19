import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import gdown
import os
import time
import requests
from typing import Tuple, Optional

# ======================
# CONFIGURATION
# ======================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

DRIVE_FOLDER_ID = "1gwGG918XTWKST--gTuMtKDxnFyPFzgXV"
MODEL_FILENAME = "final_model.keras"
MODEL_PATH = "final_model.keras"
CLASS_NAMES = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
IMAGE_SIZE = (224, 224)
MIN_CONFIDENCE = 0.7

# ======================
# MODEL DOWNLOAD UTILITIES
# ======================
def get_drive_file_id(folder_id: str, filename: str) -> Optional[str]:
    """Get file ID from Google Drive folder"""
    try:
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        response = requests.get(url)
        if response.status_code == 200:
            # This is a simplified approach - in production you might need Google Drive API
            for line in response.text.split('\n'):
                if filename in line and 'data-id' in line:
                    return line.split('data-id="')[1].split('"')[0]
    except Exception as e:
        st.error(f"Error accessing Drive folder: {str(e)}")
    return None

def download_from_drive(file_id: str, output_path: str) -> bool:
    """Download file from Google Drive with progress"""
    try:
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(
            url,
            output_path,
            quiet=False,
            resume=True
        )
        return os.path.exists(output_path)
    except Exception as e:
        st.error(f"Download failed: {str(e)}")
        return False

# ======================
# MODEL FUNCTIONS
# ======================
@st.cache_resource(show_spinner=False)
def load_model() -> Optional[tf.keras.Model]:
    """Load model with multiple fallback strategies"""
    # Try loading existing model first
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            os.remove(MODEL_PATH)  # Remove corrupted file
    
    # Download model with progress indicator
    with st.spinner("Locating model file in Google Drive..."):
        file_id = get_drive_file_id(DRIVE_FOLDER_ID, MODEL_FILENAME)
        
        if not file_id:
            st.error("Could not find model file in the specified Drive folder")
            return None
        
        st.info(f"Found model file (ID: {file_id}), starting download...")
        
        if download_from_drive(file_id, MODEL_PATH):
            try:
                return tf.keras.models.load_model(MODEL_PATH)
            except Exception as e:
                st.error(f"Model loading failed: {str(e)}")
                return None
        return None

# [Rest of your existing functions remain the same: preprocess_image, predict, 
# webcam_component, image_upload_component, and main]

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Optimized image preprocessing pipeline"""
    image = image.convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    img_array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image, model: tf.keras.Model) -> Tuple[str, float]:
    """Make prediction with performance tracking"""
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

def webcam_component(model: tf.keras.Model):
    """Real-time webcam detection component"""
    st.info("""
        **Tips for best results:**
        - Keep hand centered in frame
        - Use solid background if possible
        - Ensure good lighting conditions
    """)
    
    run = st.button("Start Webcam")
    stop = st.button("Stop Webcam")
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Please check permissions.")
            return
        
        frame_skip = 2  # Process every 3rd frame for better performance
        frame_count = 0
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Frame capture error")
                break
            
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Make prediction
            label, conf = predict(img_pil, model)
            
            # Display results
            cv2.putText(img_rgb, f"{label} ({conf:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            frame_placeholder.image(img_rgb, channels="RGB")
        
        cap.release()

def image_upload_component(model: tf.keras.Model):
    """Image upload and processing component"""
    uploaded_file = st.file_uploader(
        "Upload ASL Image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Clear images with visible hand signs work best"
    )
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            
            # Image editing interface
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            
            with col2:
                # Image adjustment controls
                brightness = st.slider("Brightness", 0.5, 1.5, 1.0, 0.1)
                contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
                
                # Apply adjustments
                enhancer = ImageEnhance.Brightness(img)
                img_adj = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(img_adj)
                img_adj = enhancer.enhance(contrast)
                st.image(img_adj, caption="Adjusted Image", use_column_width=True)
            
            # Make prediction
            with st.spinner("Analyzing..."):
                label, conf = predict(img_adj, model)
                
                # Visual result display
                confidence_color = "green" if conf >= MIN_CONFIDENCE else "orange"
                st.markdown(f"""
                <div style="padding: 15px; border-radius: 8px; background-color: #f8f9fa; margin-top: 20px;">
                    <h3 style="color: {confidence_color}; margin-bottom: 10px;">Prediction: {label}</h3>
                    <div style="background: linear-gradient(90deg, #4CAF50 {conf*100}%, #e0e0e0 {conf*100}%); 
                                height: 25px; border-radius: 5px;"></div>
                    <p style="text-align: center; margin-top: 8px; font-size: 16px;">
                        Confidence: <strong>{conf*100:.1f}%</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")

def main():
    # Configure page
    st.set_page_config(
        page_title="ASL Recognition Pro",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # App header
    st.title("🤟 Advanced ASL Recognition System")
    st.markdown("""
        Real-time American Sign Language detection using deep learning.  
        Get predictions via **webcam** or **image upload**.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("""
            Model failed to load. Please:
            1. Check your internet connection
            2. Ensure the model file exists in the Drive folder
            3. Verify the file is shared publicly
            4. Try again later
        """)
        st.markdown(f"""
            [Access Google Drive Folder](https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID})
        """)
        if st.button("Retry Loading"):
            st.rerun()
        return
    
    # Main application tabs
    tab1, tab2 = st.tabs(["🎥 Live Webcam Detection", "📷 Image Upload"])
    
    with tab1:
        webcam_component(model)
    
    with tab2:
        image_upload_component(model)
    
    # Footer
    st.markdown("---")
    st.caption("""
        **Note:** This system recognizes standard ASL alphabets and numbers (A-Z, 0-9).  
        Performance depends on image quality and lighting conditions.
    """)

if __name__ == "__main__":
    main()
