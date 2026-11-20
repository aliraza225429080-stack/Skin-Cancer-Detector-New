import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os

# Try to import tflite_runtime first (preferred for small deployments), fallback to tensorflow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

st.set_page_config(page_title="Skin Cancer Classification App", layout="centered")

st.title("🩺 Skin Cancer Classification Web App")
st.write("Upload an image of a skin lesion to analyze it using our compressed Float16 TensorFlow Lite model.")

# 1. Load Pretrained Assets
@st.cache_resource
def load_assets():
    tflite_path = 'model.tflite'
    interpreter = None
    if os.path.exists(tflite_path):
        interpreter = tflite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

    # Load Tabular preprocessing assets if needed
    scaler_path = 'scaler.pkl'
    if os.path.exists(scaler_path):
        assets = joblib.load(scaler_path)
    else:
        assets = None

    return interpreter, assets

interpreter, assets = load_assets()

if interpreter is None:
    st.error("Model file 'model.tflite' not found. Please ensure it is saved in the same directory.")
else:
    # Get input and output tensors details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 2. Upload Image UI
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)

        st.sidebar.header("Image Metadata Details")
        width, height = image.size
        mode = image.mode
        st.sidebar.write(f"**Original Width:** {width} px")
        st.sidebar.write(f"**Original Height:** {height} px")
        st.sidebar.write(f"**Color Mode:** {mode}")

        # Process prediction
        if st.button("Predict Diagnosis"):
            with st.spinner('Analyzing the image...'):
                # Preprocess image for the TFLite model (224, 224, 3)
                img_resized = image.resize((224, 224))
                if img_resized.mode != 'RGB':
                    img_resized = img_resized.convert('RGB')

                img_array = np.array(img_resized, dtype=np.float32)
                img_array = np.expand_dims(img_array, axis=0)  # Shape (1, 224, 224, 3)

                # Set input tensor
                interpreter.set_tensor(input_details[0]['index'], img_array)
                
                # Run inference
                interpreter.invoke()

                # Get prediction
                prediction_prob = interpreter.get_tensor(output_details[0]['index'])[0][0]

                st.subheader("Result & Confidence Analysis")
                if prediction_prob >= 0.5:
                    st.error(f"🚨 **Result: Malignant** (Confidence: {prediction_prob*100:.2f}%)")
                else:
                    st.success(f"✅ **Result: Benign** (Confidence: {(1 - prediction_prob)*100:.2f}%)")
