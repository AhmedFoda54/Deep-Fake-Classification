import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Decorator to load the model only once
@st.cache_resource
def load_model():
    # Load the TFLite model
    interpreter = tf.lite.Interpreter(model_path="deepfake_classifier.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Preprocess and predict
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match the model's input shape
    image = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and expand dims for batch input
    return image

def predict(image):
    image = preprocess_image(image)

    # Load the model
    interpreter = load_model()
    
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output = interpreter.get_tensor(output_details[0]['index'])

    # Return the prediction (assuming binary classification)
    return int(np.round(output[0][0]))

# Streamlit App
st.title("Deep Fake Classifier")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        result = predict(image)
        st.write(f"Prediction: {'Real' if result == 1 else 'Fake'}")
