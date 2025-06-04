import os
import gdown
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_URL = "https://www.dropbox.com/scl/fi/sssonhyqh8ocjzi6hysca/deepfake_classifier.tflite?rlkey=3wzqsgcd37ktx0u9kf3p1o75x&st=gkcr1gfr&dl=1"
MODEL_PATH = "deepfake_classifier.tflite"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model... Please wait ⏳")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Download complete ✅")

def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image
    image = np.array(image) / 255.0   # Normalize pixel values
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Add batch dim
    return image

def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])  
    score = float(output[0][0]) 
    prediction = "Real" if score > 0.5 else "Fake"
    confidence = score if prediction == "Real" else 1 - score
    return prediction, confidence
