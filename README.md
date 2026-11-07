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

The Kaggle dataset has no train/test split, so OriginalDataset was split 80/20 into dataset/train and dataset/test.

## Run
```bash
pip install -r requirements.txt
python train.py --data dataset --epochs 10      # trains the models and saves them in saved_models/
python app.py                    # open http://127.0.0.1:5000
```
Training is faster on a GPU (Google Colab works).

## Results

Trained with python train.py --data dataset --epochs 10
(seed 42, 6,400 images, 80/20 split: 5,119 train / 1,281 validation).
The validation set is the test folder; there is no separate held-out test set.
Class weights are the square root of balanced weights, to stop training
collapsing on the rare ModerateDemented class (64 images).

| Model | Validation accuracy |
|---|---|
| Custom CNN (10 epochs) | 96.96% |
| VGG16 (10 head + 10 fine-tune epochs) | 91.80% |
| MobileNetV2 (10 head + 10 fine-tune epochs) | 88.99% |

Numbers come from results.json and training_log.txt.
Results can vary slightly between runs on GPU.

## Disclaimer
Educational project only, not a medical diagnosis tool.
