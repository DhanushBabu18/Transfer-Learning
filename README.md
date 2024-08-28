# Flower Classification with Transfer Learning

This project demonstrates the use of transfer learning to classify images of flowers. Using TensorFlow and Keras, the project leverages the pre-trained ResNet50 model to classify images into five categories: daisy, dandelion, roses, sunflowers, and tulips.

## Project Overview

The goal of this project is to build a flower image classifier using transfer learning. I employ the ResNet50 architecture, which has been pre-trained on the ImageNet dataset. The pre-trained model is fine-tuned on a new dataset of flower images to perform the classification task.

### Key Components

- **Data Loading**: The flower image dataset is downloaded and preprocessed for training and validation.
- **Model Definition**: A Sequential model is created using ResNet50 as a base, with additional dense layers for classification.
- **Model Training**: The model is trained on the flower image dataset with appropriate loss functions and optimizers.
- **Evaluation**: The performance of the model is evaluated on the validation dataset, and accuracy and loss metrics are plotted.
- **Prediction**: The trained model is used to make predictions on new images.

## Dataset Description

The dataset used in this project is a collection of flower images from five different classes. 

- **Classes**:
  - Daisy
  - Dandelion
  - Roses
  - Sunflowers
  - Tulips

### Details

- **Source**: The dataset is sourced from TensorFlow's dataset repository. You can find and download the dataset [here](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz).
- **Size**: The dataset consists of a total of 3,670 images, distributed as follows:
  - **Training Set**: 2,936 images (80% of the dataset)
  - **Validation Set**: 734 images (20% of the dataset)
- **Image Dimensions**: Each image is resized to 180x180 pixels.
- **Class Distribution**: The images are roughly evenly distributed across the five flower classes, although the exact number of images per class may vary slightly.

### Example Images

You can visualize the images in the dataset using the following sample images:

- **Roses**
- **Sunflowers**
- **Tulips**
- **Dandelions**
- **Daisies**

## Requirements

To run this project, you will need the following Python libraries:

- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- OpenCV (for image processing)

You can install the necessary packages using pip:

```bash
pip install tensorflow keras matplotlib numpy pillow opencv-python
