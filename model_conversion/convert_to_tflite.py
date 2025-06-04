import tensorflow as tf
from tensorflow.keras.models import load_model

def convert_keras_to_tflite(keras_model_path, tflite_output_path):
    model = load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)
    print(f"Model converted to TFLite format and saved to {tflite_output_path}.")

if __name__ == "__main__":
    convert_keras_to_tflite("deepfake_classifier_inceptionresnet.keras", "deepfake_classifier.tflite")
