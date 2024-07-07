from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the models
sweetness_model = load_model('models/sweetness_model.keras')
sweetness_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

variety_model = load_model('models/variety_model.keras')
variety_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


def preprocess_image(img_path, target_size=(150, 150)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        try:
            processed_image = preprocess_image(file_path)

            # Predict sweetness
            sweetness_predictions = sweetness_model.predict(processed_image)
            sweetness_class = np.argmax(sweetness_predictions[0])
            sweetness_label = ["Low Sweetness", "Medium Sweetness", "High Sweetness"][sweetness_class]
            sweetness_probability = float(np.max(sweetness_predictions[0]))

            # Predict variety
            variety_predictions = variety_model.predict(processed_image)
            variety_class = np.argmax(variety_predictions[0])
            variety_labels = ["BINJAI", "E35", "N18", "PULASAN"]
            variety_label = variety_labels[variety_class]
            variety_probability = float(np.max(variety_predictions[0]))

        except Exception as e:
            return jsonify({'error': str(e)})

        return render_template('index.html',
                               sweetness_label=sweetness_label,
                               sweetness_probability=sweetness_probability,
                               variety_label=variety_label,
                               variety_probability=variety_probability,
                               image_path=file_path)


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
