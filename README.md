# Lung X-ray Classification with Deep Learning

🔬 A deep learning model to classify lung X-ray images into different categories: Normal, Viral Infection, and COVID-19.

## Overview

📝 This repository contains code for training a deep learning model to classify lung X-ray images. The model is built using TensorFlow and Keras, and it utilizes a convolutional neural network (CNN) architecture and contains code for a Streamlit app that allows users to upload lung X-ray images and classifies them using a pre-trained deep learning model.

🏥 The model is trained on a dataset of lung X-ray images and is capable of predicting whether an input X-ray image indicates a normal condition, viral infection, or COVID-19.

## Requirements

⚙️ To run this code, you need the following:

- Python 3.x
- TensorFlow 2.x
- Streamlit
- PIL (Python Imaging Library)
- Keras
- NumPy


🔢 The dataset should contain three main classes: "Normal", "Viral_Infection", and "COVID_19". The images should be split into separate folders for training and testing.

## Usage

💻 To train and evaluate the model, follow these steps:

1. Set the paths to the train and test data folders in the code.

2. Set the desired image dimensions and batch size.

3. Run the code.

4. The model will be trained on the training data and evaluated on the test data.

5. The final test accuracy will be displayed.

## Results

📊 The model's performance can be evaluated using metrics such as accuracy and loss. After training, the test accuracy will be displayed in the console.

📈 Feel free to experiment with different architectures, hyperparameters, and augmentation techniques to improve the model's performance.

## Streamlit App

🖥️ The Streamlit app provides a user-friendly interface for image classification. It allows users to upload an image, displays the uploaded image, and generates a prediction report.

📊 The prediction report includes the predicted label (Normal, COVID-19, or Pneumonia) and the corresponding probabilities for each class.

📷 The app supports various image formats, including JPEG and PNG.
