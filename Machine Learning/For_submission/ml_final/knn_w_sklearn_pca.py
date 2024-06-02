import argparse
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Learn and classify image data with a k-NN classifier.')
    parser.add_argument('-k', type=int, default=3, help='number of neighbours (if k is 0 the code may decide about proper K by itself)')
    parser.add_argument('-o', metavar='filepath', default='classification.dsv', help='path (including the filename) of the output .dsv file with the results')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('--pca', type=int, default=50, help='number of principal components for PCA (default: 50)')
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
    for i, (fname, img) in enumerate(train_images):
        dist = euclidean_distance(img, test_image)
        distances.append((dist, train_labels[i]))
    
    distances.sort(key=lambda x: x[0])
    neighbors = [label for _, label in distances[:k]]
    return max(set(neighbors), key=neighbors.count)

def main():
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    image_size = (28, 28) if '28' in args.train_path else (10, 10)
    
    train_images, train_labels_dict = load_images_and_labels(args.train_path, image_size)
    test_images, test_labels_dict = load_images_and_labels(args.test_path, image_size)

    # Extract image data and labels
    train_data = np.array([img for _, img in train_images])
    test_data = np.array([img for _, img in test_images])
    train_labels = [train_labels_dict[fname] for fname, _ in train_images]
    test_labels = [test_labels_dict[fname] for fname, _ in test_images]

    # Normalize the data
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    # Apply PCA
    pca = PCA(n_components=args.pca)
    train_data = pca.fit_transform(train_data)
    test_data = pca.transform(test_data)

    # Update train_images and test_images with PCA-transformed data
    train_images = [(fname, train_data[i]) for i, (fname, _) in enumerate(train_images)]
    test_images = [(fname, test_data[i]) for i, (fname, _) in enumerate(test_images)]

    predictions = {}
    for fname, img in test_images:
        label = knn_classify(train_images, train_labels, img, args.k)
        predictions[fname] = label
    
    accuracy = accuracy_score(test_labels, list(predictions.values()))
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    with open(args.o, 'w') as f:
        for fname, label in predictions.items():
            f.write(f"{fname}:{label}\n")

if __name__ == "__main__":
    main()
