"""Flask app: upload an MRI image, pick an algorithm, see the predicted class."""
import io, json, os
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from models import IMG_SIZE, MODEL_BUILDERS

app = Flask(__name__)
CLASS_NAMES = json.load(open("saved_models/class_names.json"))
_cache = {}

def get_model(name):
    if name not in _cache:
        _cache[name] = load_model(f"saved_models/{name}.keras")
    return _cache[name]

@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    if request.method == "POST":
        file = request.files.get("image")
        algo = request.form.get("algorithm")
        if not file or algo not in MODEL_BUILDERS:
            error = "Please choose an image and an algorithm."
        else:
            img = Image.open(io.BytesIO(file.read())).convert("RGB").resize(IMG_SIZE)
            x = np.expand_dims(np.array(img, dtype="float32"), 0)
            probs = get_model(algo).predict(x, verbose=0)[0]
            i = int(np.argmax(probs))
            result = {"label": CLASS_NAMES[i], "confidence": f"{probs[i]:.1%}", "algo": algo}
    return render_template("index.html", algorithms=list(MODEL_BUILDERS), result=result, error=error)

if __name__ == "__main__":
    app.run(debug=True)
