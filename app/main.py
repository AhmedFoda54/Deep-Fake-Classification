import streamlit as st
from PIL import Image
from app.model_utils import download_model, load_model, preprocess_image, predict

st.title("Deep Fake Image Classifier 🕵️‍♂️")

download_model()
interpreter = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction, confidence = predict(interpreter, processed_image)
        st.write(f"### **Prediction:** {prediction}")
        st.write(f"### **Confidence Score:** {confidence:.4f}")
