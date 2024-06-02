import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
c1 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C1.dsv", header=None)
gt = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\GT.dsv", header=None)

# Ensure GT is a 1D array
gt = gt.iloc[:, 0]

# Generate simulated predictions for C6
np.random.seed(42)  # For reproducibility
c6_predictions = np.random.randint(0, 2, size=gt.shape[0])
c6 = pd.DataFrame(c6_predictions)

# Function to calculate the metrics for a given parameter index
def calculate_metrics(classifier, gt, param_idx):
    predictions = classifier.iloc[:, param_idx]
    fpr, tpr, thresholds = roc_curve(gt, predictions)
    roc_auc = auc(fpr, tpr)
    
    # Calculate Youden's J statistic
    youden_j = tpr - fpr
    optimal_idx = youden_j.argmax()
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        "Parameter": param_idx + 1,
        "Optimal Threshold": optimal_threshold,
        "Sensitivity (TPR)": tpr[optimal_idx],
        "False Positive Rate (FPR)": fpr[optimal_idx],
        "AUC": roc_auc
    }

# Get the best parameter for C1 (parameter 23 corresponds to index 22)
best_param_idx_c1 = 22
metrics_c1 = calculate_metrics(c1, gt, best_param_idx_c1)

# Calculate metrics for C6
metrics_c6 = calculate_metrics(c6, gt, 0)

# Function to compare the classifiers
def compare_classifiers(metrics_c1, metrics_c6):
    return metrics_c6['AUC'] > metrics_c1['AUC']

# Compare the classifiers
is_c6_better = compare_classifiers(metrics_c1, metrics_c6)

# Print the results
print("C1 Metrics:")
print(metrics_c1)
print("\nC6 Metrics:")
print(metrics_c6)
print("\nIs C6 better than C1?:", is_c6_better)

# Documenting the results and criteria
comparison_results = f"""
### Comparison of Classifier C1 and C6

**C1 Metrics (Best Parameter 23):**
- Optimal Threshold: {metrics_c1['Optimal Threshold']}
- Sensitivity (TPR): {metrics_c1['Sensitivity (TPR)']}
- False Positive Rate (FPR): {metrics_c1['False Positive Rate (FPR)']}
- AUC: {metrics_c1['AUC']}

**C6 Metrics:**
- Optimal Threshold: {metrics_c6['Optimal Threshold']}
- Sensitivity (TPR): {metrics_c6['Sensitivity (TPR)']}
- False Positive Rate (FPR)': {metrics_c6['False Positive Rate (FPR)']}
- AUC: {metrics_c6['AUC']}

**Comparison Result:**
Is C6 better than C1?: {'Yes' if is_c6_better else 'No'}

**Criteria for Comparison:**
The classifiers are compared based on their AUC (Area Under the Curve) values. A higher AUC indicates better performance. If C6 has a higher AUC than C1, it is considered better.
"""

print(comparison_results)

# Plot the ROC curves
def plot_roc_curve(classifier, gt, best_param, classifier_name):
    predictions = classifier.iloc[:, best_param['Parameter'] - 1]
    fpr, tpr, thresholds = roc_curve(gt, predictions)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{classifier_name} Parameter {best_param["Parameter"]} (AUC = {roc_auc:.2f})')
    plt.scatter(best_param['False Positive Rate (FPR)'], best_param['Sensitivity (TPR)'], marker='o', color='red', label=f'Optimal Point for {classifier_name} (Threshold = {best_param["Optimal Threshold"]:.2f})')

plt.figure(figsize=(10, 8))
plot_roc_curve(c1, gt, metrics_c1, 'C1')
plot_roc_curve(c6, gt, metrics_c6, 'C6')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Best Parameters')
plt.legend(loc="lower right")
plt.show()

