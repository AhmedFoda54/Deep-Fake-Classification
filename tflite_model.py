import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("deepfake_classifier_inceptionresnet.keras")

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite format successfully!")
