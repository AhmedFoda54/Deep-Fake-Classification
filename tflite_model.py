import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model

# Define the CustomScaleLayer
class CustomScaleLayer(Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        # If inputs is a list, apply scaling to each element
        if isinstance(inputs, list):
            return [input_tensor * self.scale for input_tensor in inputs]
        else:
            return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config

# Register the custom layer
custom_objects = {"CustomScaleLayer": CustomScaleLayer}

# Load the model (replace with the correct path to your model)
model = load_model("C:/Users/ahmed foda/Model Deployment/InceptionResnet_DeepFake.h5", custom_objects=custom_objects)

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model converted to TFLite format successfully!")

# Define the CustomScaleLayer TEMP
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, list):  # If multiple tensors, apply scaling to each
            return [tf.multiply(tensor, self.scale) for tensor in inputs]
        else:
            return tf.multiply(inputs, self.scale)

    def compute_output_shape(self, input_shape):
        return input_shape  # Ensure output shape is the same as input

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config