import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


MODEL_PATH = "model/tea_disease_mobilenet.keras"


# Load trained model
model = load_model(MODEL_PATH)


# Must be same as training
IMG_SIZE = 224


# Same order as flow_from_directory()
class_names = [
    '1. Tea algal leaf spot',
    '2. Brown Blight',
    '3. Gray Blight',
    '7. Healthy leaf',
    '8.brown blight',
    '9. red leaf spot',
    '4. white spot'
]


def predict_disease(img_path):

    # Load image
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )


    # Convert image to array
    img_array = image.img_to_array(img)


    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )


    # Same preprocessing as training
    img_array = img_array / 255.0


    # Prediction
    predictions = model.predict(
        img_array,
        verbose=0
    )


    # Get highest probability class
    predicted_class_index = np.argmax(
        predictions[0]
    )


    # Confidence
    confidence = np.max(
        predictions[0]
    ) * 100


    # Get disease name
    disease = class_names[
        predicted_class_index
    ]


    return disease, round(float(confidence), 2)