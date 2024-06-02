import argparse
import os
import numpy as np
from PIL import Image
from collections import defaultdict
from sklearn.model_selection import train_test_split

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data with a naive bayes classifier.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-o', metavar='filepath', default='classification.dsv', help='path (including the filename) of the output .dsv file with the results')
    return parser

def load_images_and_labels(data_path, image_size=None):
    labels = {}
    images = []
    
    with open(os.path.join(data_path, 'truth.dsv'), 'r') as f:
        for line in f:
            filename, label = line.strip().split(':')
            labels[filename] = label
    
    for fname in os.listdir(data_path):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(data_path, fname)).convert('L')
            if image_size:
                img = img.resize(image_size)  # Resize image if specified
            img = np.array(img).flatten()
            images.append((fname, img))
    
    return images, labels

def train_naive_bayes(train_images, train_labels):
    label_count = defaultdict(int)
    pixel_count = defaultdict(lambda: np.zeros(256))
    for fname, img in train_images:
        label = train_labels[fname]
        label_count[label] += 1
        for pixel in img:
            pixel_count[label][pixel] += 1
    
    total_images = sum(label_count.values())
    priors = {label: count / total_images for label, count in label_count.items()}
    likelihoods = {label: (pixel_count[label] + 1) / (label_count[label] + 256) for label in label_count}
    
    return priors, likelihoods

def naive_bayes_classify(priors, likelihoods, test_image):
    posteriors = {}
    for label in priors:
        posterior = np.log(priors[label])
        for pixel in test_image:
            posterior += np.log(likelihoods[label][pixel])
        posteriors[label] = posterior
    
    return max(posteriors, key=posteriors.get)

def calculate_accuracy(predictions, labels):
    correct = sum(predictions[filename] == label for filename, label in labels.items())
    return correct / len(labels)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    image_size = (28, 28) if '28' in args.train_path else (10, 10)
    
    train_images, train_labels = load_images_and_labels(args.train_path, image_size)
    test_images, test_labels = load_images_and_labels(args.test_path, image_size)
    
    priors, likelihoods = train_naive_bayes(train_images, train_labels)
    
    predictions = {}
    for fname, img in test_images:
        label = naive_bayes_classify(priors, likelihoods, img)
        predictions[fname] = label
    
    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    with open(args.o, 'w') as f:
        for fname, label in predictions.items():
            f.write(f"{fname}:{label}\n")

if __name__ == "__main__":
    main()
