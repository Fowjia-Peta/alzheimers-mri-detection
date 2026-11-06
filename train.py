"""Train and compare the four models on the MRI dataset.

Expected folder layout (one sub-folder per class):
  dataset/train/<class_name>/*.jpg
  dataset/test/<class_name>/*.jpg
Usage: python train.py --epochs 10
"""
import argparse, json, os
import tensorflow as tf
from models import MODEL_BUILDERS, IMG_SIZE

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=32)
args = parser.parse_args()

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.data, "train"), image_size=IMG_SIZE, batch_size=args.batch, seed=42)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(args.data, "test"), image_size=IMG_SIZE, batch_size=args.batch, shuffle=False)

class_names = train_ds.class_names
print("Classes:", class_names)
os.makedirs("saved_models", exist_ok=True)
json.dump(class_names, open("saved_models/class_names.json", "w"))

AUTOTUNE = tf.data.AUTOTUNE
train_ds, test_ds = train_ds.prefetch(AUTOTUNE), test_ds.prefetch(AUTOTUNE)

results = {}
for name, build in MODEL_BUILDERS.items():
    print(f"\n=== Training {name} ===")
    model = build()
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=test_ds, epochs=args.epochs)
    loss, acc = model.evaluate(test_ds, verbose=0)
    results[name] = round(float(acc), 4)
    model.save(f"saved_models/{name}.keras")

print("\nTest accuracy per model:")
for k, v in results.items():
    print(f"  {k:14s} {v:.4f}")
json.dump(results, open("saved_models/results.json", "w"), indent=2)
