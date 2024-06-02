import argparse
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data with a k-NN classifier.')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('-k', type=int, default=3, help='number of neighbours (if k is 0 the code may decide about proper K by itself)')
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

def euclidean_distance(img1, img2):
    diff = img1 - img2
    return np.sqrt(np.sum(np.square(diff)))

def knn_classify(train_images, train_labels, test_image, k):
    distances = []
    for fname, img in train_images:
        dist = euclidean_distance(img, test_image)
        distances.append((dist, train_labels[fname]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = [label for _, label in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

def calculate_accuracy(predictions, labels):
    correct = sum(predictions[filename] == label for filename, label in labels.items())
    return correct / len(labels)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    image_size = (28, 28) if '28' in args.train_path else (10, 10)
    
    train_images, train_labels = load_images_and_labels(args.train_path, image_size)
    test_images, test_labels = load_images_and_labels(args.test_path, image_size)
    
    predictions = {}
    for fname, img in test_images:
        label = knn_classify(train_images, train_labels, img, args.k)
        predictions[fname] = label
    
    accuracy = calculate_accuracy(predictions, test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    with open(args.o, 'w') as f:
        for fname, label in predictions.items():
            f.write(f"{fname}:{label}\n")

if __name__ == "__main__":
    main()
