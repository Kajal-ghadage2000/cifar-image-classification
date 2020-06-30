import base64
import numpy as np
import io
from PIL import Image

import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array

from flask import request
from flask import jsonify
from flask import Flask
from keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

def get_model():
    global model
    model = keras.models.load_model('Model.h5')
    model._make_predict_function()
    print(" $$$$$ Model loaded sucessfully!")
    
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image=(image.astype('float32'))/255
    return image


print(" $$$$$$$$ Loading Keras model...")
get_model()



@app.route("/predict", methods=["POST"])
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))		
    processed_image = preprocess_image(image, target_size=(32, 32))
    
    prediction = model.predict(processed_image)
    a = np.argmax(prediction)

    class_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    variable = class_label[a]

    response = {
		'classpred': variable
    }

    return jsonify(response)


if __name__ == '__main__':
   app.run()
