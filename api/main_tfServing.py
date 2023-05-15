from fastapi import FastAPI, UploadFile, File
import uvicorn
from io import BytesIO
import numpy as np
import os
from PIL import Image  # image manipulation liabrary
import tensorflow as tf
import requests
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

# if any request come from local host, then its allowed.
# some security policy will not block the request after usinf cors.
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


endpoint = "http://localhost:8550/v1/models/pepper_model:predict"


# Get the absolute path to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))

model_version = max([int(i) for i in os.listdir(
    "../idg_models") if i != '.DS_Store'] + [0])

# Construct the path to the model file using the script's directory
# model_path = os.path.join(script_dir, "../saved_models/1")

model_path = os.path.join(
    script_dir, f"../idg_models/{model_version}/peppermodel.h5")

# MODEL = tf.keras.models.load_model("/Users/shubhamverma/Documents/Data Science:ML/Projects/PlantDiseaseClass/Colab_Code/pepper_code/saved_models/1")
MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ["Bacterial_spot", "Healthy"]


# reading request
@app.get("/ping")
async def ping():
    return "Hello, I am Coding"

# this function return ndarray data type
# open the image written in bytes


def read_image_file(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    # It is reading the file in async fassion. When many client requesting at same time and file is big,
    # IO operation will take lots of time. async can imporve the performance,
    # when one function call taking long time to read a file, python will suspand it and will server another request (fnction call).
    image = read_image_file(await file.read())
    img_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    predictions = response.json()["predictions"][0]

    # predictions = MODEL.predict(img_batch)
    # this will return the index if maximum prediction from array of predictions.
    index = np.argmax(predictions)
    predicted_class = CLASS_NAMES[index]
    confidence = np.max(predictions)
    return {
        "class": predicted_class,
        "confidence": float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8659)
