import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
data = load_iris()
X, y = data.data, data.target

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Use SHAP to explain predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Ensure shap_values and X have the correct shapes
shap_values = np.array(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[0]  # Use shap_values[1] for another class in case of multi-class problems

# Plot summary plot
shap.summary_plot(shap_values, X, feature_names=data.feature_names)

# Ensure the input for force_plot is correctly shaped
shap_value_for_single_instance = shap_values[0].reshape(1, -1) if shap_values.ndim == 2 else shap_values[0, 0].reshape(1, -1)
single_instance = X[0].reshape(1, -1)

# Plot SHAP values for a single prediction
shap.force_plot(explainer.expected_value[0], shap_value_for_single_instance, single_instance, feature_names=data.feature_names)
plt.show()
