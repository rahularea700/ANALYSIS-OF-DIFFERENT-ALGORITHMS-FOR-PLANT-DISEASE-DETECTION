from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained models
resnet_model_path = os.path.join("C:/Users/Rahul Singh/Desktop/your project/model/plant_disease_detection_resnet50.h5")
cnn_model_path = os.path.join("C:/Users/Rahul Singh/Desktop/your project/model/cnn_model.h5")
pinn_model_path = os.path.join("C:/Users/Rahul Singh/Desktop/your project/model/pinn_model.h5")

resnet_model = load_model(resnet_model_path)
cnn_model = load_model(cnn_model_path)
pinn_model = load_model(pinn_model_path)

# Define a dictionary to convert numeric labels back to string labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def preprocess_image(image_path, target_size=(225, 225)):
    img = load_img(image_path, target_size=target_size)
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    return x

@app.route('/')
def index():
    return render_template('new.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file and selected model from the POST request
        file = request.files['file']
        model_choice = request.form.get('model')
        
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            x = preprocess_image(file_path)

            # Select the model based on user choice
            if model_choice == 'resnet':
                predictions = resnet_model.predict(x)
            elif model_choice == 'cnn':
                predictions = cnn_model.predict(x)
            elif model_choice == 'pinn':
                predictions = pinn_model.predict(x)
            else:
                return jsonify({'error': 'Invalid model selected'}), 400

            predicted_label = labels[np.argmax(predictions)]

            # Return the result as JSON
            return jsonify({'predicted_label': predicted_label})
        else:
            return jsonify({'error': 'No file uploaded'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
