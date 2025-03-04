# Traffic Sign Recognition with CNN + Tsetlin Machine

This project combines Convolutional Neural Networks (CNN) with Tsetlin Machine learning for traffic sign classification. It demonstrates how CNNs can be used for feature extraction before applying Tsetlin Machine learning for the final classification.

## Project Overview

The notebook (`Tsetlin_With_CNN.ipynb`) implements a hybrid approach to traffic sign recognition:

1. **CNN for Feature Extraction**: Uses a convolutional neural network to extract meaningful features from traffic sign images
2. **Tsetlin Machine for Classification**: Takes the CNN-extracted features, binarizes them, and uses a Tsetlin Machine classifier for the final classification

This hybrid approach leverages the strengths of both methods - CNN's ability to extract hierarchical spatial features from images and Tsetlin Machine's interpretable rule-based classification.

## Dataset

The German Traffic Sign Recognition Benchmark (GTSRB) dataset is used, which includes:
- 34,799 training images
- 4,410 validation images
- 12,630 test images

Each image is a 32×32 RGB color image of a traffic sign belonging to one of 43 classes.

## Implementation Steps

1. **Data Loading**: Loads data from pickle files and CSV mappings
2. **Data Preprocessing**: Converts images to grayscale and binarizes them
3. **CNN Architecture**: 
   - 3 Convolutional layers with ReLU activation
   - MaxPooling after each convolutional layer
   - A dense feature layer with 256 units
   - Output layer with 43 units (one for each traffic sign class)
4. **Feature Extraction**: The CNN is first trained on the raw images, then the feature layer is used to extract features
5. **Feature Binarization**: The extracted features are binarized for use with the Tsetlin Machine
6. **Tsetlin Machine Training**: A TMClassifier is trained on the binarized features with early stopping
7. **Evaluation**: The model is evaluated on the test set to measure its accuracy

## Results

The hybrid CNN + Tsetlin Machine approach achieves approximately 90% accuracy on the test set, demonstrating the effectiveness of this combined approach for traffic sign recognition.

## Requirements

- TensorFlow/Keras for CNN implementation
- TMU (Tsetlin Machine Library)
- NumPy, Pandas for data manipulation
- Matplotlib for visualization
- OpenCV for image preprocessing

## Usage

The notebook is self-contained and can be run in a Jupyter environment with all the required dependencies installed. The data files should be placed in the appropriate paths as specified in the notebook.

## Model Hyperparameters

### CNN
- 3 Convolutional layers (32, 64, 128 filters)
- 3×3 kernels with 'same' padding
- 2×2 max pooling
- 256 neurons in the feature extraction layer
- Adam optimizer

### Tsetlin Machine
- 5,000 clauses for pattern recognition
- Threshold (T) = 50
- Specificity (s) = 10.0
- Clause dropout = 0.1