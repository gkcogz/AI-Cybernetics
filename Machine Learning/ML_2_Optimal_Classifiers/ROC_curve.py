import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# Load the datasets
c1 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C1.dsv", header=None)
c2 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C2.dsv", header=None)
c3 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C3.dsv", header=None)
c4 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C4.dsv", header=None)
c5 = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\C5.dsv", header=None)
gt = pd.read_csv("C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\ML_2_Optimal_Classifiers\\Classifiers\\GT.dsv", header=None)

# Function to calculate sensitivity and false positive rate
def calculate_metrics(predictions, ground_truth):
    tn, fp, fn, tp = confusion_matrix(ground_truth, predictions).ravel()
    sensitivity = tp / (tp + fn)  # Sensitivity (Recall)
    false_positive_rate = fp / (fp + tn)  # False Positive Rate
    return sensitivity, false_positive_rate

# Calculate metrics for each classifier
metrics = {}
for classifier, name in zip([c1, c2, c3, c4, c5], ['c1', 'c2', 'c3', 'c4', 'c5']):
    predictions = classifier.iloc[:, 0]
    sensitivity, false_positive_rate = calculate_metrics(predictions, gt.iloc[:, 0])
    metrics[name] = {"Sensitivity": sensitivity, "False Positive Rate": false_positive_rate}

# Display the metrics
metrics_df = pd.DataFrame(metrics).T
print(metrics_df)

# Plot ROC curves
plt.figure(figsize=(10, 8))

for classifier, name in zip([c1, c2, c3, c4, c5], ['c1', 'c2', 'c3', 'c4', 'c5']):
    predictions = classifier.iloc[:, 0]
    fpr, tpr, _ = roc_curve(gt.iloc[:, 0], predictions)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
