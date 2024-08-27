import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

model = load_model('jute_pest_detection_model.h5')

class_labels = [
    'Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 
    'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 
    'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 
    'Pod Borer', 'Scopula Emissaria', 'Termite', 
    'Termite odontotermes (Rambur)', 'Yellow Mite'
]

def main():
    st.title("Jute Pest Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        filepath = os.path.join('uploads', uploaded_file.name)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        st.image(filepath, caption='Uploaded Image', use_column_width=True)
        
        prediction = predict_image(filepath)
        st.write(f"Prediction: {prediction}")

def predict_image(filepath):
    img = load_img(filepath, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

if not os.path.exists('uploads'):
    os.makedirs('uploads')

if __name__ == "__main__":
    main()
