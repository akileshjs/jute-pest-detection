from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os

app = Flask(__name__)

model = load_model('jute_pest_detection_model.h5')

class_labels = [
    'Beet Armyworm', 'Black Hairy', 'Cutworm', 'Field Cricket', 
    'Jute Aphid', 'Jute Hairy', 'Jute Red Mite', 'Jute Semilooper', 
    'Jute Stem Girdler', 'Jute Stem Weevil', 'Leaf Beetle', 'Mealybug', 
    'Pod Borer', 'Scopula Emissaria', 'Termite', 
    'Termite odontotermes (Rambur)', 'Yellow Mite'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filepath = os.path.join('uploads', file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            return render_template('result.html', prediction=prediction, image_path=filepath)
    return render_template('index.html')

def predict_image(filepath):
    img = load_img(filepath, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return class_labels[predicted_class]

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
