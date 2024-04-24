# README

## Overview
This repository contains code for building and training a convolutional neural network (CNN) using TensorFlow and Keras to classify images of cats and dogs. The dataset used for training consists of images of cats and dogs stored in the directory 'drive/MyDrive/samecatanddog/'.

## Requirements
- TensorFlow
- Matplotlib
- NumPy
- PIL

## Instructions
1. **Mounting Google Drive**: The code mounts Google Drive to access the dataset stored in the specified directory.

2. **Loading and Preprocessing Data**: The dataset is loaded using `tf.keras.utils.image_dataset_from_directory()` function. It is split into training and validation sets with a split ratio of 80:20. The images are resized to a standard size of 50x50 pixels and normalized.

3. **Visualization of Training Images**: A sample of images from the training dataset is visualized using Matplotlib.

4. **Model Building**: A CNN model is built using the Sequential API of TensorFlow and Keras. It consists of convolutional layers, max-pooling layers, and fully connected layers.

5. **Model Compilation**: The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy metric.

6. **Model Training**: The model is trained on the training dataset for 10 epochs with validation on the validation dataset.

7. **Visualization of Feature Maps**: Feature maps from selected layers of the trained model are visualized for interpretation and analysis.

## Results
- The model achieves 100% accuracy on both the training and validation datasets after 10 epochs, indicating effective learning and generalization.
- Feature maps from selected layers are visualized to understand how the model learns to detect patterns and features in the images.

## Conclusion
- The CNN model successfully classifies images of cats and dogs with high accuracy, demonstrating the effectiveness of deep learning for image classification tasks.
- Visualization of feature maps provides insights into the inner workings of the model and helps understand its learning process.
