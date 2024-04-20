from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#initialize Flask application
app = Flask(__name__)
app.secret_key = 'secret key'

#directory for uploaded files
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#load model
model = tf.keras.models.load_model('/Users/pranavgupta/Documents/Synopsys_2024/dataset/IIModelCNNFFT.h5')  # Adjust the path to your model

#check if the file is allowed based on its extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#preprocess image
def process_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims
def compute_fft(image_path):
    img = image.load_img(image_path, color_mode='grayscale', target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.fft.fft2(img_array)
    img_array = np.fft.fftshift(img_array)
    img_array = np.abs(img_array)
    img_array = np.log(img_array + 1)
    img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            processed_image = process_image(filepath)
            fft_image = compute_fft(filepath)
            ###os.remove(filepath)###
            prediction = model.predict([processed_image, fft_image])
            if prediction[0][0] < 0.5:
                result = 'Real'
            else:
                result = 'Fake'
            return render_template('index.html', prediction=result)
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)