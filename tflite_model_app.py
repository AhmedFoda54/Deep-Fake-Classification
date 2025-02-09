import streamlit as st
import gdown
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Dropbox link for the model
MODEL_URL = "https://www.dropbox.com/scl/fi/sssonhyqh8ocjzi6hysca/deepfake_classifier.tflite?rlkey=3wzqsgcd37ktx0u9kf3p1o75x&st=gkcr1gfr&dl=1"
MODEL_PATH = "deepfake_classifier.tflite"

# Function to download the model if it's not downloaded priviously
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
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = "Real" if output[0][0] > 0.5 else "Fake"
    confidence = output[0][0]
    return prediction, confidence

# Streamlit App UI
st.title("Deep Fake Image Classifier 🕵️‍♂️")
download_model()
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction, confidence = predict(processed_image)
        st.write(f"### **Prediction:** {prediction}")
        st.write(f"### **Confidence Score:** {confidence:.4f}")
