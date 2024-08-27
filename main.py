# Import necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

# Load the trained model
model = load_model('jute_pest_detection_model.h5')

# Class labels for predictions
class_labels = [
    'Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 
    'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 
    'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 
    'Pod Borer', 'Scopula Emissaria', 'Termite', 
    'Termite odontotermes (Rambur)', 'Yellow Mite'
]

# Define the Streamlit app
def main():
    st.title("Jute Pest Detection")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        # Save the uploaded image to a temporary file
        filepath = os.path.join('uploads', uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Display the uploaded image
        st.image(filepath, caption='Uploaded Image', use_column_width=True)
        
        # Predict the image
        prediction = predict_image(filepath)
        st.write(f"Prediction: {prediction}")

def predict_image(filepath):
    # Load and preprocess the image
    img = load_img(filepath, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    # Make prediction
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

# Create the uploads folder if it does not exist
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Run the Streamlit app
if __name__ == "__main__":
    main()
