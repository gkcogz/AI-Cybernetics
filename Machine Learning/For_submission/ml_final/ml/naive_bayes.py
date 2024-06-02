import argparse
import os
import numpy as np
from PIL import Image
from collections import defaultdict

def setup_arg_parser():
    """
    Sets up the argument parser for the command line arguments.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Learn and classify image data with a naive bayes classifier.')
    parser.add_argument('-o', metavar='filepath', default='classification.dsv', help='path (including the filename) of the output .dsv file with the results')
    parser.add_argument('train_path', type=str, help='path to the training data directory')
    parser.add_argument('test_path', type=str, help='path to the testing data directory')
    parser.add_argument('--pca', type=int, default=50, help='number of principal components for PCA (default: None)')
    return parser

def load_images_and_labels(data_path, image_size=None):
    """
    Loads images and their corresponding labels from a specified directory.

    Args:
        data_path (str): Path to the directory containing the images and the truth.dsv file.
        image_size (tuple, optional): The desired size to resize the images. Defaults to None.

    Returns:
        list: A list of tuples where each tuple contains the filename and the image data as a flattened array.
        dict: A dictionary mapping filenames to labels.
    """
    labels = {}
    images = []
    
    # Load labels from truth.dsv if it exists
    label_file = os.path.join(data_path, 'truth.dsv')
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                filename, label = line.strip().split(':')
                labels[filename] = label
    
    # Load images from the directory
    for fname in os.listdir(data_path):
        if fname.endswith('.png'):
            img = Image.open(os.path.join(data_path, fname)).convert('L')
            if image_size:
                img = img.resize(image_size)  # Resize image if specified
            img = np.array(img).flatten() / 255.0  # Normalize pixel values
            images.append((fname, img))
    
    print(f"Loaded {len(images)} images from {data_path}")
    return images, labels

def pca(data, num_components):
    """
    Applies Principal Component Analysis (PCA) to reduce the dimensionality of the data.

    Args:
        data (np.array): The data to reduce, with each row being an image and each column a pixel.
        num_components (int): The number of principal components to retain.

    Returns:
        np.array: The transformed data after PCA.
        np.array: The mean of the original data.
        np.array: The eigenvectors used for transformation.
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    eigenvector_subset = sorted_eigenvectors[:, :num_components]
    pca_data = np.dot(centered_data, eigenvector_subset)
    return pca_data, mean, eigenvector_subset

def train_gaussian_naive_bayes(train_images, train_labels):
    """
    Trains a Gaussian Naive Bayes classifier.

    Args:
        train_images (list): List of tuples containing the filename and the flattened image data for training images.
        train_labels (list): List of labels corresponding to the training images.

    Returns:
        dict: Priors for each label.
        dict: Means of the pixel values for each label.
        dict: Variances of the pixel values for each label.
    """
    label_count = defaultdict(int)
    pixel_sum = defaultdict(lambda: np.zeros(train_images[0][1].shape))
    pixel_sq_sum = defaultdict(lambda: np.zeros(train_images[0][1].shape))

    for i, (fname, img) in enumerate(train_images):
        label = train_labels[i]
        label_count[label] += 1
        pixel_sum[label] += img
        pixel_sq_sum[label] += img ** 2

    total_images = sum(label_count.values())
    priors = {label: count / total_images for label, count in label_count.items()}
    means = {label: pixel_sum[label] / label_count[label] for label in label_count}
    variances = {label: (pixel_sq_sum[label] / label_count[label]) - (means[label] ** 2) for label in label_count}
    
    return priors, means, variances

def gaussian_probability(x, mean, var):
    """
    Computes the Gaussian probability of a value.

    Args:
        x (float): The value.
        mean (float): The mean of the distribution.
        var (float): The variance of the distribution.

    Returns:
        float: The Gaussian probability of the value.
    """
    eps = 1e-6  # to avoid division by zero
    coefficient = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
    exponential = np.exp(-(x - mean) ** 2 / (2 * var + eps))
    return coefficient * exponential

def gaussian_naive_bayes_classify(priors, means, variances, test_image):
    """
    Classifies a test image using the Gaussian Naive Bayes algorithm.

    Args:
        priors (dict): Priors for each label.
        means (dict): Means of the pixel values for each label.
        variances (dict): Variances of the pixel values for each label.
        test_image (np.array): Flattened array of pixel values for the test image.

    Returns:
        str: The predicted label for the test image.
    """
    posteriors = {}
    for label in priors:
        posterior = np.log(priors[label])
        for i in range(len(test_image)):
            posterior += np.log(gaussian_probability(test_image[i], means[label][i], variances[label][i]))
        posteriors[label] = posterior
    return max(posteriors, key=posteriors.get)

def main():
    """
    Main function to execute the Gaussian Naive Bayes classification.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    image_size = (28, 28) if '28' in args.train_path else (10, 10)
    
    # Load training and testing data
    train_images, train_labels_dict = load_images_and_labels(args.train_path, image_size)
    test_images, _ = load_images_and_labels(args.test_path, image_size)  # Ignore test labels

    # Convert lists to numpy arrays and normalize
    train_data = np.array([img for _, img in train_images])
    test_data = np.array([img for _, img in test_images])
    train_labels = [train_labels_dict[fname] for fname, _ in train_images]

    train_data = train_data / 255.0
    test_data = test_data / 255.0

    # Apply PCA if specified
    if args.pca is not None:
        train_data, mean, eigenvectors = pca(train_data, args.pca)
        test_data = np.dot(test_data - mean, eigenvectors)

    # Update images with PCA-transformed data
    train_images = [(fname, train_data[i]) for i, (fname, _) in enumerate(train_images)]
    test_images = [(fname, test_data[i]) for i, (fname, _) in enumerate(test_images)]

    # Train the Gaussian Naive Bayes model
    priors, means, variances = train_gaussian_naive_bayes(train_images, train_labels)
    
    # Classify test images
    predictions = {}
    for fname, img in test_images:
        label = gaussian_naive_bayes_classify(priors, means, variances, img)
        predictions[fname] = label
    
    # Save results to file
    with open(args.o, 'w') as f:
        for fname, label in predictions.items():
            f.write(f"{fname}:{label}\n")

    print(f"Classification results saved to {args.o}")

if __name__ == "__main__":
    main()
