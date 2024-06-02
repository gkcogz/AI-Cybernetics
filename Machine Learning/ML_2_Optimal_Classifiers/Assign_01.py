import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt

# Load the data
c1 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C1.dsv", header=None)
gt = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\GT.dsv", header=None)

# Ensure they are 1D arrays
c1 = c1.iloc[:, 0]  # Assuming the predictions are in the first column
gt = gt.iloc[:, 0]  # Assuming the ground truth is in the first column

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(gt, c1)

# Calculate the Youden's J statistic
youden_j = tpr - fpr
optimal_idx = youden_j.argmax()
optimal_threshold = thresholds[optimal_idx]

# Optimal metrics
optimal_fpr = fpr[optimal_idx]
optimal_tpr = tpr[optimal_idx]
roc_auc = auc(fpr, tpr)

# Display the results
optimal_metrics = {
    "Optimal Threshold": optimal_threshold,
    "Sensitivity (TPR)": optimal_tpr,
    "False Positive Rate (FPR)": optimal_fpr,
    "AUC": roc_auc
}

# Document the results in a DataFrame
optimal_metrics_df = pd.DataFrame([optimal_metrics])
print(optimal_metrics_df)

# Plot the ROC curve with the optimal point
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'C1 (AUC = {roc_auc:.2f})')
plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Point (Threshold = {optimal_threshold:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
