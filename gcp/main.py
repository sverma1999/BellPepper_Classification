from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np


BUCKET_NAME = 'vshubham-tf-models'
CLASS_NAMES = ["Bacterial Spot", "Healthy"]
model = None


def download_blob(bucket_name, source_blob_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_blob_name)


def predict(request):
    global model
    if model is None:
        download_blob(
            BUCKET_NAME,
            "models/peppermodel.h5",
            "/tmp/peppermodel.h5"
        )
        model = tf.keras.models.load_model("/tmp/peppermodel.h5")
    image = request.files["file"]
    image = np.array(
        Image.open(image).convert("RGB").resize((256, 256)) # image resizing
    )
    image = image/255
    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    index = np.argmax(predictions[0]) # this will return the index if maximum prediction from array of predictions.
    predicted_class = CLASS_NAMES[index]
    confidence = round(100 * (np.max(predictions[0])),2)
    return {
        "class": predicted_class,
        "confidence": confidence
    }




