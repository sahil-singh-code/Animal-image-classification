# Multi-Class Animal Image Classification Using CNN

## Overview

This project classifies animal images into 10 categories using a custom Convolutional Neural Network (CNN). A Random Forest classifier serves as a baseline to compare traditional machine learning with deep learning for image classification.

## Dataset

- **Dataset:** Animals-10
- **Total images:** 26,179
- **Number of classes:** 10
- **Classes:** Butterfly, Cat, Chicken, Cow, Dog, Elephant, Horse, Sheep, Spider, and Squirrel
- **Data split:** 70% training, 15% validation, 15% testing

## Methodology

1. Explored the dataset, image classes, and class distribution.
2. Split the dataset into training, validation, and test sets using stratified sampling.
3. Built a Random Forest baseline using flattened 32 × 32 RGB images.
4. Developed a CNN using 128 × 128 RGB images, convolutional layers, max pooling, fully connected layers, and dropout.
5. Used the Adam optimizer, categorical probability outputs, and early stopping based on validation loss.
6. Evaluated both models using test accuracy and macro F1-score.
7. Generated a classification report, normalized confusion matrix, and training accuracy/loss curves.
8. Saved the trained CNN, class-name mapping, and evaluation outputs.

## Results

| Model | Test Accuracy | Macro F1-score |
|---|---:|---:|
| Random Forest | 41.58% | 34.63% |
| Custom CNN | 64.86% | 60.49% |

The CNN achieved a **23.28 percentage-point improvement** in test accuracy over the Random Forest baseline, based on the displayed rounded accuracy values.

The models use different input resolutions, so this comparison represents a practical baseline rather than a resolution-matched experiment.

## Visualizations

### CNN Normalized Confusion Matrix

The confusion matrix shows the proportion of predictions for each true animal class, helping identify which classes the model distinguishes well and which it confuses.

### CNN Training Curves

The training curves show training and validation accuracy and loss across epochs. The widening accuracy gap and rising validation loss toward the end suggest some overfitting.

## Technologies

- Python
- TensorFlow and Keras
- NumPy and Pandas
- Matplotlib
- Scikit-learn
- Pillow

## Project Structure

```text
animal-image-classification/
├── project.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── animal_cnn.keras
│   └── class_names.json
└── results/
    ├── cnn_classification_report.csv
    ├── cnn_confusion_matrix.png
    ├── cnn_training_curves.png
    └── model_comparison.csv
```

The dataset is excluded from the repository because of its size.

## How to Run

1. Clone or download this repository.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and extract the Animals-10 dataset into the appropriate local directory.
4. Update `DATA_DIR` in `project.ipynb` to point to your dataset's `raw-img` directory.
5. Run the notebook cells in order to reproduce preprocessing, training, and evaluation.

For prediction using the saved model, load `models/animal_cnn.keras` and read the corresponding class order from `models/class_names.json`. Retraining is not necessary for ordinary inference.

## Notes

The reported metrics correspond to the current experiment and may vary with changes to the data split, preprocessing, or training configuration.
