def calculate_accuracy(predictions_file, truth_file):
    with open(predictions_file, 'r') as pred_f, open(truth_file, 'r') as truth_f:
        predictions = {line.split(':')[0]: line.split(':')[1].strip() for line in pred_f}
        truths = {line.split(':')[0]: line.split(':')[1].strip() for line in truth_f}

    correct = sum(predictions[fname] == label for fname, label in truths.items())
    total = len(truths)
    accuracy = correct / total
    return accuracy

# Example usage:
predictions_file = "C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\For_submission\\ml_final\\classification_1000_10.dsv"
truth_file = "C:\\Users\\gokce\\OneDrive\\Desktop\\CTU Prague\\Cybernetics & AI\\Assignments\\Coding_Assignments\\Machine Learning\\train_1000_10\\test\\truth.dsv"
accuracy = calculate_accuracy(predictions_file, truth_file)
print(f"Accuracy: {accuracy * 100:.2f}%")
