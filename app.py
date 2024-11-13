from flask import Flask, request, render_template, jsonify
from keras.models import load_model
from keras.applications import VGG16
from keras.models import Model
import numpy as np
import pickle
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

app = Flask(__name__)

cnn_model = load_model('./fish_custom_cnn_model.keras')
with open('./lr_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)

classes = ['betta fish', 'corydoras fish', 'discus fish', 'flowerhorn fish', 
           'goldfish', 'guppy', 'neocaridina', 'neon fish', 
           'oscar fish', 'platy fish']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'model' not in request.form:
        return jsonify({"error": "No file or model selection found"}), 400

    file = request.files['file']
    model_choice = request.form['model']
    image = Image.open(file).resize((224, 224))
    image_array = np.array(image)

    if model_choice == 'cnn':
        image_array = mobilenet_preprocess(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        predictions = cnn_model.predict(image_array)
        predicted_class = classes[np.argmax(predictions)]

    elif model_choice == 'lr':
        image_array = vgg_preprocess(image_array)
        image_array = np.expand_dims(image_array, axis=0)
        features = feature_extractor.predict(image_array)
        features_flattened = features.reshape(1, -1)  
        prediction = lr_model.predict(features_flattened)
        predicted_class = classes[int(prediction[0])]

    else:
        return jsonify({"error": "Invalid model selection"}), 400

    return jsonify({"prediction": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
