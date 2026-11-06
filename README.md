# Alzheimer's Disease Detection from MRI using Deep Learning

AI-ML virtual internship project (AICTE-EduSkills, supported by AWS Academy, Jul-Sep 2022).

## What it does
Classifies a brain MRI scan into 4 stages: **Non Demented, Very Mild Demented, Mild Demented, Moderate Demented**.
Three models are trained and compared, and a Flask web app lets a user upload an MRI image, choose an algorithm
and view the predicted class.

- Models: Custom CNN, VGG16 (transfer learning),  MobileNetV2 (transfer learning)
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
python train.py --epochs 10      # trains the models and saves them in saved_models/
python app.py                    # open http://127.0.0.1:5000
```
Training is faster on a GPU (Google Colab works).

## Results
Validation accuracy (4-class, 6,400 MRI images, 80/20 split, 128x128):

| Model | Val accuracy |
|---|---|
| Custom CNN | 85.4% |
| VGG16 (fine-tuned) | 81.8% |
| MobileNetV2 (fine-tuned) | 79.9% |

## Disclaimer
Educational project only, not a medical diagnosis tool.
