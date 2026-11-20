import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import joblib
import os

st.set_page_config(page_title="Skin Cancer Classification App", layout="centered")

st.title("🩺 Skin Cancer Classification Web App")
st.write("Upload an image of a skin lesion to analyze it using our trained Convolutional Neural Network (CNN).")

# 1. Load Pretrained Assets
@st.cache_resource
def load_assets():
    # Load CNN Keras model
    model_path = 'model.tflite'
    if os.path.exists(model_path):
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
    else:
        model=None
    # Load Tabular preprocessing assets if needed
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        assets = joblib.load(scaler_path)
    else:
        assets = None
        
    return model, assets

model, assets = load_assets()

if model is None:
    st.error("Model file 'CNN_MODELS.keras' not found. Please ensure it is saved in the same directory.")
else:
    # 2. Upload Image UI
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        # Prepare structured/tabular inputs for completeness if requested
        st.sidebar.header("Image Metadata Details")
        width, height = image.size
        mode = image.mode
        st.sidebar.write(f"**Original Width:** {width} px")
        st.sidebar.write(f"**Original Height:** {height} px")
        st.sidebar.write(f"**Color Mode:** {mode}")
        
        # Process prediction button
        if st.button("Predict Diagnosis"):
            with st.spinner('Analyzing the image...'):
                # Preprocess image for the CNN
                img_resized = image.resize((224, 224))
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')
                
                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)
                
                # Make prediction
                prediction_prob = model.predict(img_array)[0][0]
                
                st.subheader("Result & Confidence Analysis")
                if prediction_prob >= 0.5:
                    st.error(f"🚨 **Result: Malignant** (Confidence: {prediction_prob*100:.2f}%)")
                else:
                    st.success(f"✅ **Result: Benign** (Confidence: {(1 - prediction_prob)*100:.2f}%)")
