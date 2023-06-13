import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model_upd.h5')

# Define the class labels
class_labels = ['Normal', 'COVID-19', 'Pneumonia']

# Function to preprocess the image
def preprocess_image(image):
    # Convert the image to RGB
    image = image.convert('RGB')
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert the PIL image to a NumPy array
    image_array = np.array(image)
    # Reshape the array to match the input shape of the model
    image_array = image_array.reshape(1, 224, 224, 3)
    # Normalize the image array
    image_array = image_array / 255.0
    return image_array


# Function to make a prediction
def make_prediction(image):
    # Preprocess the image
    image_array = preprocess_image(image)
    # Make the prediction using the model
    predictions = model.predict(image_array)
    # Get the predicted class label
    predicted_label = class_labels[np.argmax(predictions)]
    # Get the prediction probabilities
    probabilities = [round(float(p) * 100, 2) for p in predictions[0]]
    # Create the prediction report
    report = {
        'Predicted Label': predicted_label,
        'Probabilities': dict(zip(class_labels, probabilities))
    }
    return report

# Streamlit app
def main():
    st.title("Lung X-ray Classification")
    st.write("Upload an image of a lung X-ray and the model will classify it.")

    # File upload
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Make a prediction and display the report
        if st.button('Classify'):
            report = make_prediction(image)
            st.subheader("Prediction Report")
            st.write("Predicted Label:", report['Predicted Label'])
            st.write("Probabilities:")
            for label, prob in report['Probabilities'].items():
                st.write(f"{label}: {prob}%")

if __name__ == '__main__':
    main()