"""Train and compare Custom CNN, VGG16 and MobileNetV2 on the 4-class Alzheimer MRI dataset.

Usage:
    python train.py --data dataset --epochs 10
Expected folders: dataset/train/<class>/*.jpg  and  dataset/test/<class>/*.jpg
(the test folder is used as the validation set).
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from models import (IMG_SIZE, MODEL_FILES, build_cnn, build_mobilenetv2,
                    build_vgg16)


def load_data(data_dir, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"), image_size=IMG_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=True, seed=42)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"), image_size=IMG_SIZE,
        batch_size=batch_size, label_mode="int", shuffle=False)
    class_names = train_ds.class_names
    labels = np.concatenate([y.numpy() for _, y in train_ds])
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    class_weight = {int(i): float(np.sqrt(w)) for i, w in enumerate(weights)}
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), class_names, class_weight


def compile_model(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])


def fit(model, train_ds, val_ds, epochs, class_weight):
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weight)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="saved_models")
    args = p.parse_args()

    tf.keras.utils.set_random_seed(42)
    os.makedirs(args.out, exist_ok=True)
    train_ds, val_ds, class_names, cw = load_data(args.data, args.batch_size)
    with open(os.path.join(args.out, "class_names.json"), "w") as f:
        json.dump(class_names, f)
    results = {}

    # 1) Custom CNN
    cnn = build_cnn()
    compile_model(cnn, 1e-3)
    fit(cnn, train_ds, val_ds, args.epochs, cw)
    results["cnn"] = float(cnn.evaluate(val_ds, verbose=0)[1])
    cnn.save(os.path.join(args.out, MODEL_FILES["cnn"]))

    # 2) MobileNetV2: train the new head, then fine-tune the last 30 layers
    mn, base = build_mobilenetv2()
    compile_model(mn, 1e-3)
    fit(mn, train_ds, val_ds, args.epochs, cw)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(mn, 1e-4)
    fit(mn, train_ds, val_ds, args.epochs, cw)
    results["mobilenetv2"] = float(mn.evaluate(val_ds, verbose=0)[1])
    mn.save(os.path.join(args.out, MODEL_FILES["mobilenetv2"]))

    # 3) VGG16: train the new head, then fine-tune the last conv block
    vgg, vbase = build_vgg16()
    compile_model(vgg, 1e-3)
    fit(vgg, train_ds, val_ds, args.epochs, cw)
    vbase.trainable = True
    for layer in vbase.layers[:-4]:
        layer.trainable = False
    compile_model(vgg, 1e-5)
    fit(vgg, train_ds, val_ds, args.epochs, cw)
    results["vgg16"] = float(vgg.evaluate(val_ds, verbose=0)[1])
    vgg.save(os.path.join(args.out, MODEL_FILES["vgg16"]))

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    for name, acc in results.items():
        print(f"{name}: validation accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
