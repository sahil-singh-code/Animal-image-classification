# Multi-Class Animal Image Classification Using CNN

## Overview

An image-classification project that identifies animals across 10 categories using a custom Convolutional Neural Network (CNN). A Random Forest classifier is used as a baseline for performance comparison.

## Dataset

- Dataset: Animals-10
- Total images: 26,179
- Classes: Dog, Horse, Elephant, Butterfly, Chicken, Cat, Cow, Sheep, Spider, and Squirrel
- Split: 70% training, 15% validation, 15% testing

## Methodology

1. Explored image classes and class distribution.
2. Resized CNN inputs to 128 × 128 RGB images and normalized pixel values to [0, 1].
3. Trained a Random Forest baseline using flattened 32 × 32 images.
4. Built a CNN with Conv2D, MaxPooling2D, Dense layers, and Dropout.
5. Used validation data and Early Stopping to monitor training.
6. Evaluated performance using test accuracy, macro F1-score, classification reports, and confusion matrices.

## Results

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Random Forest | 41.58% | 34.63% |
| Custom CNN | 64.86% | 60.49% |

The CNN improved test accuracy by 23.27 percentage points over the Random Forest baseline.

Note: The models use different input resolutions, so this is a practical baseline comparison rather than a controlled resolution-matched experiment.

## Technologies

Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn, Pillow.

## Project Structure

- `01_data_exploration.ipynb` — data exploration, preprocessing, model training, and evaluation.
- `models/` — saved CNN and class-name mapping.
- `results/` — model comparison, classification report, confusion matrix, and training curves.

## How to Run

1. Install the dependencies listed in `requirements.txt`.
2. Download the Animals-10 dataset and extract its class folders.
3. Update the dataset path in the notebook.
4. Run the notebook cells in order.

The reported metrics are from the current experiment and may vary if the data split, preprocessing, or training configuration changes.