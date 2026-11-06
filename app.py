"""Flask app: upload an MRI image, choose a model and see the predicted class."""
import io
import json
import os

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image

import models  # registers the custom VGG preprocessing layer

app = Flask(__name__)
MODEL_DIR = "saved_models"
_cache = {}


def class_names():
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        return json.load(f)


def get_model(key):
    if key not in _cache:
        path = os.path.join(MODEL_DIR, models.MODEL_FILES[key])
        _cache[key] = tf.keras.models.load_model(path)
    return _cache[key]


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    if request.method == "POST":
        key = request.form.get("algorithm", "cnn")
        file = request.files.get("image")
        if key not in models.MODEL_FILES:
            error = "Please choose a valid algorithm."
        elif not file or file.filename == "":
            error = "Please upload an MRI image."
        else:
            try:
                img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(models.IMG_SIZE)
                x = np.expand_dims(np.array(img, dtype="float32"), 0)
                probs = get_model(key).predict(x, verbose=0)[0]
                names = class_names()
                idx = int(np.argmax(probs))
                result = {
                    "label": names[idx],
                    "confidence": f"{probs[idx] * 100:.1f}",
                    "model": models.MODEL_LABELS[key],
                }
            except Exception as exc:  # unreadable image or missing saved model
                error = f"Could not run prediction: {exc}"
    return render_template("index.html", labels=models.MODEL_LABELS, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=True)
