"""Four model definitions used in the project: Modified CNN, VGG16, AlexNet-style, MobileNet."""
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, MobileNet

IMG_SIZE = (128, 128)
INPUT_SHAPE = IMG_SIZE + (3,)
NUM_CLASSES = 4   # Non Demented, Very Mild, Mild, Moderate


def modified_cnn():
    return models.Sequential([
        layers.Input(INPUT_SHAPE),
        layers.Rescaling(1./255),
        layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"), layers.MaxPooling2D(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="modified_cnn")


def alexnet():
    """Smaller AlexNet-style network (5 conv layers + 3 dense layers)."""
    return models.Sequential([
        layers.Input(INPUT_SHAPE),
        layers.Rescaling(1./255),
        layers.Conv2D(96, 11, strides=4, activation="relu"), layers.MaxPooling2D(3, strides=2),
        layers.Conv2D(256, 5, padding="same", activation="relu"), layers.MaxPooling2D(3, strides=2),
        layers.Conv2D(384, 3, padding="same", activation="relu"),
        layers.Conv2D(384, 3, padding="same", activation="relu"),
        layers.Conv2D(256, 3, padding="same", activation="relu"), layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(512, activation="relu"), layers.Dropout(0.5),
        layers.Dense(256, activation="relu"), layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="alexnet")


def _transfer(base_cls, name):
    base = base_cls(include_top=False, weights="imagenet", input_shape=INPUT_SHAPE)
    base.trainable = False   # transfer learning: freeze pre-trained layers
    return models.Sequential([
        layers.Input(INPUT_SHAPE),
        layers.Rescaling(1./255),
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=name)


def vgg16():
    return _transfer(VGG16, "vgg16")


def mobilenet():
    return _transfer(MobileNet, "mobilenet")


MODEL_BUILDERS = {
    "modified_cnn": modified_cnn,
    "vgg16": vgg16,
    "alexnet": alexnet,
    "mobilenet": mobilenet,
}
