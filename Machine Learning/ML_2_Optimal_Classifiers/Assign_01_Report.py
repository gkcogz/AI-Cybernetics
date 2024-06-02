import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

# Load the data
c1 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C1.dsv", header=None)
gt = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\GT.dsv", header=None)

# Ensure ground truth is a 1D array
gt = gt.iloc[:, 0]

# Initialize a list to store metrics for each parameter
all_metrics = []

# Loop through each parameter (column) in c1
for param_idx in range(c1.shape[1]):
    predictions = c1.iloc[:, param_idx]
    
    # Calculate the ROC curve
    fpr, tpr, thresholds = roc_curve(gt, predictions)
    
    # Calculate the Youden's J statistic
    youden_j = tpr - fpr
    optimal_idx = youden_j.argmax()
    optimal_threshold = thresholds[optimal_idx]

    # Optimal metrics
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]
    roc_auc = auc(fpr, tpr)

    # Store metrics
    metrics = {
        "Parameter": param_idx,
        "Optimal Threshold": optimal_threshold,
        "Sensitivity (TPR)": optimal_tpr,
        "False Positive Rate (FPR)": optimal_fpr,
        "AUC": roc_auc
    }
    all_metrics.append(metrics)

# Convert the metrics to a DataFrame
metrics_df = pd.DataFrame(all_metrics)

# Find the parameter with the highest AUC
best_param_idx = metrics_df['AUC'].idxmax()
best_metrics = metrics_df.loc[best_param_idx]

# Plot the ROC curve for the best parameter
best_predictions = c1.iloc[:, best_param_idx]
fpr, tpr, thresholds = roc_curve(gt, best_predictions)
roc_auc = auc(fpr, tpr)
optimal_threshold = best_metrics['Optimal Threshold']
optimal_fpr = best_metrics['False Positive Rate (FPR)']
optimal_tpr = best_metrics['Sensitivity (TPR)']

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, label=f'C1 Parameter {best_param_idx} (AUC = {roc_auc:.2f})')
plt.scatter(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Point (Threshold = {optimal_threshold:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
roc_curve_path = "roc_curve_best_param.png"
plt.savefig(roc_curve_path)
plt.show()

# Generate the table as an image
fig, ax = plt.subplots(figsize=(10, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=best_metrics.values.reshape(1, -1), colLabels=best_metrics.index, cellLoc='center', loc='center')
table_image_path = "optimal_metrics_table_best_param.png"
plt.savefig(table_image_path)
plt.show()

# Combine images into a PDF
images = [Image.open(roc_curve_path), Image.open(table_image_path)]
pdf_path = "Optimal_Parameter_Report_C1.pdf"
images[0].save(pdf_path, save_all=True, append_images=images[1:])

print(f"Report saved as {pdf_path}")


