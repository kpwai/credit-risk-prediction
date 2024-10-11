# Gradient Boosting Classifier with SHAP Explainability for German Credit Risk Dataset

This project uses a **Gradient Boosting Classifier** to predict credit risk (good or bad credit) using the **German Credit Dataset**. The model is explained using **SHAP** to visualize feature importance and understand the model's predictions at both global and local levels.

## Dataset

The German Credit dataset is obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)). It contains information on 1000 loan applicants with the following features:
- Various socio-demographic and financial information.
- The target label: **Credit Risk**, where 1 indicates "Good Credit" and 2 indicates "Bad Credit".

## Model

A **Gradient Boosting Classifier** is used for predicting whether a loan applicant has a good or bad credit risk. The model is trained on 80% of the data and evaluated on 20% of the data.

## Techniques Used

- **Gradient Boosting Classifier**: A powerful ensemble learning method that combines multiple decision trees to enhance prediction accuracy.
- **Standardization**: Features are standardized to improve model performance.
- **SHAP**: Model explainability is provided using **SHapley Additive exPlanations (SHAP)** to understand feature importance.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Python script:
   ```bash
      python GB_german_Shap.py
   ```
   The model will train on the dataset and output accuracy metrics, confusion matrix, and SHAP visualizations.

## Results
The Gradient Boosting Classifier achieved the following accuracy on the test data:

Accuracy on Test Data: (the actual accuracy will be shown in the output)
Confusion matrix and classification report will also be displayed to evaluate model performance.

## Explainability
The project provides explainability using SHAP:

SHAP Summary Plot: Visualizes global feature importance, showing which features are most important in predicting credit risk.
SHAP Force Plot: Visualizes the contribution of each feature to an individual prediction.
SHAP Waterfall Plot: Provides a detailed breakdown of feature contributions to a specific prediction.
