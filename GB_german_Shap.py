# Install necessary libraries
# !pip install shap scikit-learn matplotlib seaborn

# Import Libraries
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the German Credit dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
column_names = [
    "Status of existing checking account", "Duration in month", "Credit history", "Purpose",
    "Credit amount", "Savings account/bonds", "Present employment since", "Installment rate in percentage of disposable income",
    "Personal status and sex", "Other debtors / guarantors", "Present residence since", "Property",
    "Age in years", "Other installment plans", "Housing", "Number of existing credits at this bank",
    "Job", "Number of people being liable to provide maintenance for", "Telephone", "foreign worker", "Credit risk"
]

data = pd.read_csv(url, delim_whitespace=True, names=column_names)

# Encode categorical features
categorical_features = data.select_dtypes(include=['object']).columns
le = LabelEncoder()
data[categorical_features] = data[categorical_features].apply(le.fit_transform)

# Split dataset into features and labels (Credit risk is the target column, 1=Good, 2=Bad)
X = data.drop('Credit risk', axis=1)
y = data['Credit risk']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Model Accuracy on Test Data:", accuracy_score(y_test, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=['Good Credit', 'Bad Credit'], yticklabels=['Good Credit', 'Bad Credit'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Good Credit', 'Bad Credit']))

# --- SHAP Explainability ---
# Create a SHAP explainer for Gradient Boosting Classifier
explainer = shap.Explainer(model, X_train)

# Get SHAP values for the test data
shap_values = explainer(X_test)

# SHAP Summary Plot (global feature importance)
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# SHAP Force Plot for a single instance (local explainability)
i = 0  # Index of the test instance to explain
shap.initjs()
shap.force_plot(shap_values[i].base_values, shap_values[i].values, X_test[i])

# SHAP Waterfall plot for a single instance (local explainability)
shap.waterfall_plot(shap_values[i])

# SHAP Bar Plot (global feature importance)
shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=X.columns)