"""Model builders: custom CNN, VGG16 and MobileNetV2 (transfer learning)."""
import tensorflow as tf
from tensorflow.keras import layers, models

IMG_SIZE = (128, 128)
INPUT_SHAPE = IMG_SIZE + (3,)
NUM_CLASSES = 4
MODEL_FILES = {
    "cnn": "cnn.keras",
    "vgg16": "vgg16.keras",
    "mobilenetv2": "mobilenetv2.keras",
}
MODEL_LABELS = {
    "cnn": "Custom CNN",
    "vgg16": "VGG16",
    "mobilenetv2": "MobileNetV2",
}


@tf.keras.utils.register_keras_serializable(package="alz")
class VGGPreprocess(layers.Layer):
    """VGG16 preprocessing (RGB->BGR, subtract ImageNet mean) as a saveable layer."""

    def call(self, x):
        return tf.keras.applications.vgg16.preprocess_input(x)

    def compute_output_shape(self, input_shape):
        return input_shape


def build_cnn():
    return models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cnn")


def build_vgg16():
    base = tf.keras.applications.VGG16(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        VGGPreprocess(),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="vgg16")
    return model, base


def build_mobilenetv2():
    base = tf.keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet")
    base.trainable = False
    model = models.Sequential([
        layers.Input(shape=INPUT_SHAPE),
        layers.Rescaling(1.0 / 127.5, offset=-1),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="mobilenetv2")
    return model, base
