# Deep Fake Image Classifier 🕵️‍♂️

A **Streamlit** web app that uses a **TensorFlow Lite** model to classify images as either **Real** or **Fake**. This project helps detect deep fake images using deep learning pretrained model optimized for lightweight performance.

---

## 🗕️ **Features**

- Upload an image and detect whether it's **Real** or **Fake**.
- Displays the **confidence score** of the prediction.
- Lightweight and fast inference with **TensorFlow Lite**.
- Simple and user-friendly **Streamlit** interface.

---

## 🛠️ **Installation**

1. **Clone the repository**

```bash
git clone https://github.com/AhmedFoda54/Deep-Fake-Classification.git
```

2. **Create and activate a virtual environment (optional but recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. **Install the required packages**

```bash
pip install -r requirements.txt
```

---

## 🚀 **Running the App**

1. **Run the Streamlit app:**

```bash
streamlit run tflite_model_app.py
```

2. **Open your browser** and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

---

## 📂 **Project Structure**

```
deepfake-classifier/
|-- tflite_model_app.py                      # Main Streamlit application file
|-- tflite_model.py                      # To convert the .h5/keras model to .tflite model
|-- deepfake_classifier.tflite  # TensorFlow Lite model file (downloaded automatically)
|-- requirements.txt            # List of required Python packages
|-- README.md                   # Project documentation
```

---

## 🎯 **Model Information**

- The model is hosted on **Dropbox** and automatically downloaded when you run the app.
- It uses a **TensorFlow Lite** format optimized for fast, efficient performance.
---

## 🌐 **How to Use**

1. **Upload an Image:** Click on the file uploader and select an image (`.jpg`, `.jpeg`, or `.png`).
2. **View the Image:** The app displays the uploaded image.
3. **Click Predict:** Press the "Predict" button to classify the image.
4. **View Results:** The app will show whether the image is **Real** or **Fake**, along with a confidence score.

---

## 📊 **Dependencies**

Make sure the following packages are installed (included in `requirements.txt`):

- `streamlit`
- `tensorflow`
- `numpy`
- `pillow`
- `gdown`
- `os`

---

## 💼 **Author**

- **Ahmed Foda**  
  Data Science & Coding Instructor Enthusiast  
  [LinkedIn](https://www.linkedin.com/in/ahmed-abdelghany-2b4621253/) | [Email](mailto:s-ahmed.foda@zewailcity.edu.eg)

---

## 💍 **License**

This project is licensed under the **MIT License**.

---

## 🔗 **References**

1. Heidari, A., Navimipour, N. J., Dag, H., & Unal, M. (2023). Deepfake detection using deep learning methods: A systematic and comprehensive review. Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, 14(2). https://doi.org/10.1002/widm.1520

2. Abir, W. H., et al. (2022). Detecting deepfake images using deep learning techniques and explainable AI methods. Intelligent Automation & Soft Computing, 35(2), 2151–2169. https://doi.org/10.32604/iasc.2023.029653

3. Salih, A. M., Raisi-Estabragh, Z., Galazzo, I. B., Radeva, P., Petersen, S. E., Lekadir, K., & Menegaz, G. (2024). A Perspective on Explainable Artificial Intelligence Methods: SHAP and LIME. arXiv. https://arxiv.org/abs/2305.02012



