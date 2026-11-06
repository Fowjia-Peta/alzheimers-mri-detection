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
python train.py --epochs 5      # trains the models and saves them in saved_models/
python app.py                    # open http://127.0.0.1:5000
```
Training is faster on a GPU (Google Colab works).

## Results

Trained on the Alzheimer's 4-class MRI dataset (Kaggle: pashacqu/alzheimers-dataset-4-class-of-images, OriginalDataset). The data was split 80/20 into train and validation sets. Each model was trained for 5 epochs on a Colab T4 GPU.

| Model | Validation Accuracy |
|---|---|
| Custom CNN | 95.00% |
| MobileNetV2 | 87.82% |
| VGG16 | 75.10% |

Note: results are from a single 5-epoch run with a random 80/20 split, so numbers may vary slightly between runs.

## Disclaimer
Educational project only, not a medical diagnosis tool.
