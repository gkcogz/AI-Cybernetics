import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the data
c1 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C1.dsv", header=None)
c2 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C2.dsv", header=None)
c3 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C3.dsv", header=None)
c4 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C4.dsv", header=None)
c5 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C5.dsv", header=None)
gt = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\GT.dsv", header=None)

# Ensure GT is a 1D array
gt = gt.iloc[:, 0]

# Function to calculate the metrics for each parameter
def calculate_metrics(classifier, gt):
    optimal_metrics = []
    for i in range(classifier.shape[1]):
        predictions = classifier.iloc[:, i]
        fpr, tpr, thresholds = roc_curve(gt, predictions)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Youden's J statistic
        youden_j = tpr - fpr
        optimal_idx = youden_j.argmax()
        optimal_threshold = thresholds[optimal_idx]
        
        # Append the metrics
        optimal_metrics.append({
            "Parameter": i + 1,
            "Optimal Threshold": optimal_threshold,
            "Sensitivity (TPR)": tpr[optimal_idx],
            "False Positive Rate (FPR)": fpr[optimal_idx],
            "AUC": roc_auc
        })
        
    return pd.DataFrame(optimal_metrics)

# Calculate metrics for each classifier
optimal_metrics_c1 = calculate_metrics(c1, gt)
optimal_metrics_c2 = calculate_metrics(c2, gt)
optimal_metrics_c3 = calculate_metrics(c3, gt)
optimal_metrics_c4 = calculate_metrics(c4, gt)
optimal_metrics_c5 = calculate_metrics(c5, gt)

# Select the best parameter for each classifier
best_param_c1 = optimal_metrics_c1.loc[optimal_metrics_c1['AUC'].idxmax()]
best_param_c2 = optimal_metrics_c2.loc[optimal_metrics_c2['AUC'].idxmax()]
best_param_c3 = optimal_metrics_c3.loc[optimal_metrics_c3['AUC'].idxmax()]
best_param_c4 = optimal_metrics_c4.loc[optimal_metrics_c4['AUC'].idxmax()]
best_param_c5 = optimal_metrics_c5.loc[optimal_metrics_c5['AUC'].idxmax()]

# Combine the best parameters into a single DataFrame
best_params = pd.DataFrame([best_param_c1, best_param_c2, best_param_c3, best_param_c4, best_param_c5], index=['C1', 'C2', 'C3', 'C4', 'C5'])

# Plot ROC curve for the best parameter of each classifier
plt.figure(figsize=(10, 8))

def plot_roc_curve(classifier, gt, best_param, label):
    parameter_idx = int(best_param['Parameter']) - 1
    predictions = classifier.iloc[:, parameter_idx]
    fpr, tpr, _ = roc_curve(gt, predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{label} Parameter {best_param["Parameter"]} (AUC = {roc_auc:.2f})')
    plt.scatter(best_param['False Positive Rate (FPR)'], best_param['Sensitivity (TPR)'], marker='o', color='red', label=f'Optimal Point for {label} (Threshold = {best_param["Optimal Threshold"]:.2f})')

plot_roc_curve(c1, gt, best_param_c1, 'C1')
plot_roc_curve(c2, gt, best_param_c2, 'C2')
plot_roc_curve(c3, gt, best_param_c3, 'C3')
plot_roc_curve(c4, gt, best_param_c4, 'C4')
plot_roc_curve(c5, gt, best_param_c5, 'C5')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Best Parameters')
plt.legend(loc="lower right")
roc_curve_best_param_path = "roc_curve_best_param.png"
plt.savefig(roc_curve_best_param_path)
plt.show()

# Generate the table as an image
fig, ax = plt.subplots(figsize=(12, 2))  # Adjust size as needed
ax.axis('tight')
ax.axis('off')
ax.table(cellText=best_params.values, colLabels=best_params.columns, rowLabels=best_params.index, cellLoc='center', loc='center')
table_image_path = "optimal_metrics_table_best_param.png"
plt.savefig(table_image_path)
plt.show()

# Print the best parameters
print(best_params)
