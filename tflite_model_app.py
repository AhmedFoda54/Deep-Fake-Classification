import streamlit as st
import gdown
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Google Drive link for the model (Change this to your actual model ID)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1A3UQqjIwNr_qDD0PymIPNBtSI0hySF26"
MODEL_PATH = "deepfake_classifier.tflite"

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... Please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.write("Download complete ✅")

# Load the TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Expand dims to match model input
    return image

# Run inference with TFLite model
def predict(image):
    interpreter = load_model()
    
    # Get input/output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Run inference
    interpreter.invoke()

    # Get output tensor
    output = interpreter.get_tensor(output_details[0]['index'])

    # Binary classification (Assuming output is a single probability score)
    prediction = "Real" if output[0][0] > 0.5 else "Fake"
    confidence = output[0][0]

    return prediction, confidence

# Streamlit App UI
st.title("Deep Fake Image Classifier 🕵️‍♂️")

# Download model if not present
download_model()

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        # Preprocess image and make prediction
        processed_image = preprocess_image(image)
        prediction, confidence = predict(processed_image)

        # Display results
        st.write(f"### **Prediction:** {prediction}")
        st.write(f"### **Confidence Score:** {confidence:.4f}")

