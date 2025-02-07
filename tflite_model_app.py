import streamlit as st
import gdown
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# Google Drive link for the model (Change this to your actual model ID)
MODEL_URL = "MODEL_URL = "https://drive.google.com/uc?export=download&id=1A3UQqjIwNr_qDD0PymIPNBtSI0hySF26"
MODEL_PATH = "deepfake_classifier.tflite"

# Function to download the model
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.write("Downloading model... Please wait ⏳")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.write("Download complete ✅")
        except Exception as e:
            st.error(f"Error downloading model: {e}")
    
    # Check if the file exists after download
    if not os.path.exists(MODEL_PATH):
        st.error("Model download failed. Please check the link or try again.")
    else:
        st.success("Model is ready!")

# Load the TFLite model
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=os.path.abspath(MODEL_PATH))  # Use absolute path
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the input image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Expand dims to match model input
    return image

# Run inference with TFLite model
def predict(image):
    interpreter = load_model()
    if interpreter is None:
        return "Error", 0.0

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
    confidence = output[0][0] if prediction == "Real" else 1 - output[0][0]

    return prediction, confidence

# Streamlit App UI
st.title("Deep Fake Image Classifier 🕵️‍♂️")

# Download model if not present
download_model()

# Check if model exists before proceeding
if os.path.exists(MODEL_PATH):
    st.success(f"Model loaded from: {os.path.abspath(MODEL_PATH)}")
else:
    st.error("Model file is missing. Please check the download link and try again.")

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

        if prediction == "Error":
            st.error("Failed to run the prediction. Please check the model file.")
        else:
            # Display results
            st.write(f"### **Prediction:** {prediction}")
            st.write(f"### **Confidence Score:** {confidence:.4f}")
