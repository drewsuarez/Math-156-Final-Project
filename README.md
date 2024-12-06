# Math 156 Final Project

This repository contains the code and documentation for the Math 156 Final Project, focusing on using Convolutional Neural Networks (CNNs) for car image classification.

## Overview

The goal of this project is to classify car images into one of seven classes using a CNN. This classification task addresses real-world applications such as vehicle identification in surveillance systems. The project involves building a CNN model from scratch, training it on a labeled dataset, and evaluating its performance.

## Dataset

The dataset used is the [Car Images Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data). It consists of 4,165 images across 7 car classes:
- Audi
- Hyundai Creta
- Mahindra Scorpio
- Rolls Royce
- Swift
- Tata Safari
- Toyota Innova

### Preprocessing Steps
- Images were resized to \(128 \times 128\) pixels.
- Pixel values were normalized to the \([0, 1]\) range.
- An 80/20 train-test split was applied.

## Model Architecture

The CNN model was built using TensorFlow and includes:
- **Convolutional Layers:** Extract features from input images.
- **Pooling Layers:** Reduce spatial dimensions while retaining important features.
- **Dropout Layer:** Prevent overfitting.
- **Fully Connected Layer:** Perform the final classification with a softmax activation function.

### Key Training Parameters
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Epochs:** 10
- **Batch Size:** 32

## Results

The model achieved the following:
- **Training Accuracy:** 95.78%
- **Validation Accuracy:** 67.11%

### Observations
- Overfitting was observed after epoch 6.
- Class imbalance affected performance, with some classes like Rolls Royce having lower accuracy.

## Future Improvements
- Use balanced datasets to address class imbalance.
- Implement early stopping to prevent overfitting.
- Explore more advanced architectures like ResNet or transfer learning.
- Apply additional data augmentation techniques.

## Main References
- Kaggle Dataset: [Car Images Dataset](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)
- TensorFlow Documentation: [Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/cnn)

---


