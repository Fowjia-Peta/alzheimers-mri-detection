# Alzheimer's Disease Detection from MRI using Deep Learning

AI-ML virtual internship project (AICTE-EduSkills, supported by AWS Academy, Jul-Sep 2022).

## What it does
Classifies a brain MRI scan into 4 stages: **Non Demented, Very Mild Demented, Mild Demented, Moderate Demented**.
Four models are trained and compared, and a Flask web app lets a user upload an MRI image, choose an algorithm
and view the predicted class.

- Models: Modified CNN, VGG16 (transfer learning), AlexNet-style CNN, MobileNet (transfer learning)
- Stack: Python, Keras, TensorFlow, NumPy, Flask
- Pre-processing: images resized to 128x128 and rescaled

## Dataset
Use a public Alzheimer MRI dataset (for example the Kaggle "Alzheimer's Dataset (4 class of Images)").
Arrange it as:
```
dataset/train/<class_name>/*.jpg
dataset/test/<class_name>/*.jpg
```
The dataset is not included in this repository.

## Run
```bash
pip install -r requirements.txt
python train.py --epochs 10      # trains all 4 models, saves them in saved_models/
python app.py                    # open http://127.0.0.1:5000
```
Training is faster on a GPU (Google Colab works).

## Results
Add your own test accuracies here after training (see `saved_models/results.json`).

## Disclaimer
Educational project only, not a medical diagnosis tool.
