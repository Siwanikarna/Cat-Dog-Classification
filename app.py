import os
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

model = load_model(os.path.join(STATIC_FOLDER, 'cat-dog.h5'))

# call model to predict an image
def api(full_path):
    data = image.load_img(full_path, target_size=(150, 150))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data /= 255.

    predicted = model.predict(data)
    return predicted

# home page
@app.route('/')
def home():
    return render_template('index.html')

# procesing uploaded file and predict it
# procesing uploaded file and predict it
@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'image' not in request.files:
            # No file part in the request
            return render_template('index.html', error_message='No file chosen. Please select an image.')

        file = request.files['image']
        
        # If the user does not select a file, the browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', error_message='No file chosen. Please select an image.')
        
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)

        indices = {0: 'Cat', 1: 'Dog', 2: 'Invasive carcinomar', 3: 'Normal'}
        result = api(full_name)

        predicted_class = np.argmax(result)
        accuracy = round(result[0][predicted_class] * 100, 2)
        label = indices[predicted_class]

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    app.run(debug=True)
