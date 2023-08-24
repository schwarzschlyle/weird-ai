import io
import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import base64


app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')


# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"

        file = request.files['file']

        if file.filename == '':
            return "No selected file"

        if file and allowed_file(file.filename):
            image_stream = io.BytesIO(file.read())
            image = Image.open(image_stream).convert("RGB")  # Convert to RGB mode
            image = image.resize((224, 224))  # Resize image to match MobileNetV2 input size
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = preprocess_input(image_array)
            image_array = tf.expand_dims(image_array, 0)  # Add batch dimension

            predictions = model.predict(image_array)
            decoded_predictions = decode_predictions(predictions)[0]  # Remove .numpy()

            # Convert image to Base64
            image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

            return render_template('result.html', image_base64=image_base64, predictions=decoded_predictions)


    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
