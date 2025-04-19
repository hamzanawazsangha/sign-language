import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import gdown
import os
import time
from typing import Tuple, Optional

# Configuration Constants
MODEL_URL = "https://drive.google.com/uc?id=1UVHX3ePXl89Aeg6XxPg4QnyboGJ1SywJ"
MODEL_PATH = "final_sign.keras"
BACKUP_MODEL_URL = "https://storage.googleapis.com/your-backup-bucket/final_sign.keras"  # Add your backup URL
CLASS_NAMES = [
    '1', '10', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'Blank', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space'
]
IMAGE_SIZE = (224, 224)
MIN_CONFIDENCE = 0.7  # Minimum confidence threshold for reliable predictions

def setup_environment():
    """Configure environment settings for optimal performance"""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info messages
    tf.get_logger().setLevel('ERROR')

@st.cache_resource(show_spinner=False)
def load_model() -> Optional[tf.keras.Model]:
    """Load model with multiple fallback strategies"""
    # First try loading existing model
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            st.warning(f"Found corrupted model file. Redownloading... Error: {str(e)}")
            os.remove(MODEL_PATH)
    
    # Model download with progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def download_progress(current, total, width=80):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Downloading: {current/(1024*1024):.1f}MB / {total/(1024*1024):.1f}MB")
    
    # Try primary download source
    try:
        status_text.text("Starting download from Google Drive...")
        gdown.download(
            MODEL_URL,
            MODEL_PATH,
            quiet=True,
            fuzzy=True,
            resume=True,
            progress=download_progress
        )
    except Exception as e:
        st.warning(f"Primary download failed. Trying backup source... Error: {str(e)}")
        try:
            status_text.text("Starting download from backup server...")
            gdown.download(
                BACKUP_MODEL_URL,
                MODEL_PATH,
                quiet=True,
                resume=True,
                progress=download_progress
            )
        except Exception as e:
            st.error(f"All download attempts failed: {str(e)}")
            return None
    
    # Verify download
    if not os.path.exists(MODEL_PATH):
        st.error("Download completed but file not found. Please try again.")
        return None
    
    progress_bar.empty()
    status_text.empty()
    
    # Load the downloaded model
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Optimized image preprocessing pipeline"""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize with anti-aliasing
    image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    
    # Convert to array and normalize
    img_array = tf.keras.utils.img_to_array(image) / 255.0
    
    # Expand dimensions for batch prediction
    return np.expand_dims(img_array, axis=0)

def predict(image: Image.Image, model: tf.keras.Model) -> Tuple[str, float]:
    """Enhanced prediction with confidence thresholding"""
    img_array = preprocess_image(image)
    
    # Warm-up model (first prediction is slower)
    if not hasattr(predict, '_warmed_up'):
        model.predict(img_array)
        predict._warmed_up = True
    
    # Measure prediction time
    start_time = time.time()
    preds = model.predict(img_array, verbose=0)
    inference_time = time.time() - start_time
    
    # Get results
    class_idx = np.argmax(preds)
    confidence = float(np.max(preds))
    label = CLASS_NAMES[class_idx]
    
    # Low confidence handling
    if confidence < MIN_CONFIDENCE:
        label = f"Low Confidence ({label})"
    
    # Display performance metrics
    st.sidebar.metric("Inference Time", f"{inference_time*1000:.1f}ms")
    st.sidebar.metric("Confidence Threshold", f"{MIN_CONFIDENCE*100:.0f}%")
    
    return label, confidence

def webcam_detection(model: tf.keras.Model):
    """Optimized webcam processing with frame skipping"""
    st.info("""
        **Tips for better detection:**
        - Keep hand centered in frame
        - Use solid background if possible
        - Ensure good lighting
    """)
    
    run = st.button("Start Webcam", key="start_webcam")
    stop = st.button("Stop Webcam", key="stop_webcam")
    frame_placeholder = st.empty()
    
    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not access webcam. Please check permissions.")
            return
        
        frame_count = 0
        fps = 0
        prev_time = time.time()
        
        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("Frame capture error")
                break
            
            # Process every 3rd frame for better performance
            frame_count += 1
            if frame_count % 3 != 0:
                continue
            
            # Flip and convert color space
            frame = cv2.flip(frame, 1)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Prediction
            label, conf = predict(img_pil, model)
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Display info
            cv2.putText(img_rgb, f"{label} ({conf:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            cv2.putText(img_rgb, f"FPS: {fps:.1f}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            frame_placeholder.image(img_rgb, channels="RGB")
        
        cap.release()

def image_upload_detection(model: tf.keras.Model):
    """Enhanced image upload with preview and edit options"""
    uploaded_file = st.file_uploader(
        "Upload ASL Image",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Clear images work best. Crop to hand if possible."
    )
    
    if uploaded_file:
        try:
            img = Image.open(uploaded_file)
            
            # Image preview with edit options
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Original Image", use_column_width=True)
            
            with col2:
                # Simple image adjustments
                brightness = st.slider("Adjust Brightness", 0.5, 1.5, 1.0, 0.1)
                contrast = st.slider("Adjust Contrast", 0.5, 1.5, 1.0, 0.1)
                
                # Apply adjustments
                from PIL import ImageEnhance
                enhancer = ImageEnhance.Brightness(img)
                img_adj = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(img_adj)
                img_adj = enhancer.enhance(contrast)
                st.image(img_adj, caption="Adjusted Image", use_column_width=True)
            
            # Prediction on adjusted image
            with st.spinner("Analyzing..."):
                label, conf = predict(img_adj, model)
                
                # Visual confidence indicator
                confidence_color = "green" if conf >= MIN_CONFIDENCE else "orange"
                st.markdown(f"""
                <div style="padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
                    <h3 style="color: {confidence_color}; margin-bottom: 5px;">Prediction: {label}</h3>
                    <div style="background: linear-gradient(90deg, #4CAF50 {conf*100}%, #f0f0f0 {conf*100}%); 
                                height: 20px; border-radius: 3px;"></div>
                    <p style="text-align: center; margin-top: 5px;">Confidence: {conf*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Image processing error: {str(e)}")

def main():
    """Main application layout and logic"""
    setup_environment()
    
    # App Header
    st.set_page_config(
        page_title="ASL Recognition Pro",
        page_icon="🤟",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤟 Advanced ASL Recognition")
    st.markdown("""
        Real-time American Sign Language detection using deep learning.  
        For best results, ensure clear visibility of hand signs.
    """)
    
    # Model loading with retry option
    model_placeholder = st.empty()
    with model_placeholder.container():
        model = load_model()
    
    if model is None:
        st.error("Model failed to load. Please try again or contact support.")
        if st.button("Retry Model Loading"):
            model_placeholder.empty()
            with model_placeholder.container():
                model = load_model()
                st.experimental_rerun()
        return
    
    # Main application tabs
    tab1, tab2 = st.tabs(["🎥 Live Detection", "🖼️ Image Analysis"])
    
    with tab1:
        webcam_detection(model)
    
    with tab2:
        image_upload_detection(model)
    
    # Footer
    st.markdown("---")
    st.caption("""
        **Note:** This system recognizes standard ASL alphabets and numbers.  
        Performance depends on image quality, lighting, and background clarity.
    """)

if __name__ == "__main__":
    main()
