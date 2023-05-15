# **Bell Pepper_Classification**

## Problem Statement

Bell pepper is one of the most important crops globally, and bell pepper plants are susceptible to various diseases, which can lead to significant economic loss for farmers. Traditional methods for disease detection are often time-consuming and require expert knowledge. The aim of this project is to develop a mobile application that can quickly and accurately detect bell pepper plant diseases using image classification with deep learning, to help farmers identify and treat diseases in their crops.

## Solution Proposed

The proposed solution is to use deep learning techniques, specifically convolutional neural networks (CNNs), to classify images of bell pepper plants as either healthy or diseased. The model will be trained on a dataset of labeled images of bell pepper plants, using techniques such as data augmentation to increase the size of the dataset and improve the model's accuracy. The trained model will be deployed on a mobile application, which farmers can use to take a picture of a bell pepper plant and get an instant diagnosis of whether the plant is healthy or diseased. The application will be built using React JS and React Native for the frontend, with a backend server using FastAPI and TensorFlow Serving for ML ops. The model will also be optimized using techniques such as quantization and TensorFlow Lite to improve its performance on mobile devices.

## Tech Stack Used

1. Python
2. FastAPI
3. TensorFlow Serving
4. Machine learning algorithms
5. TensorFlow
6. Front End: JavaScript, CSS, HTML, React JS, React Native

## Infrastructure Required.

1. Google Cloud Platform (GCP) account
2. Access to Google Cloud Functions (GCF) for deployment
3. Labeled dataset of images of bell pepper plants
4. Mobile device or simulator to test the application

## How to run?

Before we run the project, make sure you have Google Cloud account to access the service like Google Cloud Platform (GCP) and Google Cloud Functions (GCF).

### Step 1: Clone the repository

```bash
git clone https://github.com/sverma1999/BellPepper_Classification.git
```

### Step 2- Create a conda environment after opening the repository

```bash
conda create -n pepper_dl python=3.8.15 -y
```

```bash
conda activate pepper_dl
```

### Step 3 - Install the requirements

```bash
pip install -r training/requirements.txt
pip install -r api/requirements.txt
```

# Training the Model

1. Download the data from [kaggle](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
2. Only keep folders related to Bell Pepper.
3. Run Jupyter Notebook in Browser.

```bash
jupyter notebook
```

4. Open training/model-training-second.ipynb in Jupyter Notebook.
5. Update all the paths.
6. Run all the Cells one by one.
7. Copy the model generated and save it with the version number in the models folder.

# Test the API locally using Postman

1. Run the API locally using FastAPI

```bash
cd  api
python main.py
```

Go to Postman application and send a POST request to http://localhost:8659/predict with the image in the body.

You should get a response like this:

```json
{
  "class": "Bacterial_spot",
  "confidence": 0.9955774545669556
}
```

2. Run the API locally using Docker for tf serving and FastAPI

   ### Why tensorflow serving?

   TensorFlow Serving is a key tool in the machine learning deployment pipeline that allows you to easily and efficiently serve your models at scale with high performance and reliability.

First, run the docker image of tensorflow serving

```bash
cd BellPepper_Classification
docker run -t --rm -p 8550:8550 -v <full path to BellPepper_Classification>:/BellPepper_Classification tensorflow/serving --rest_api_port=8550 --model_config_file=/BellPepper_Classification/models.config
```

then, run the FastAPI

```python
cd  api
python main_tfServing.py
```

Go to Postman application and send a POST request to http://localhost:8659/predict with the image in the body.

You should get a response like this:

```json
{
  "class": "Bacterial_spot",
  "confidence": 0.9955774545669556
}
```

# Setup for ReactJS

1. Install Nodejs
2. Install NPM
3. Install dependencies

```bash
cd frontend
npm install --from-lock-json
npm audit fix
```

4. Copy .env.example as .env.
5. Change API url in .env.

# Running the Frontend

1. Get inside frontend folder

```bash
cd frontend
```

2. Copy the .env.example as .env and update REACT_APP_API_URL to API URL if needed.
3. Run the frontend

```bash
npm run start
```

# Setup for React-Native app

1. Go to the [React Native environment setup](https://reactnative.dev/docs/environment-setup), then select React Native CLI Quickstart tab.
2. Install dependencies

```bash
cd mobile-app
yarn install
```

### Only for mac users

```bash
cd ios && pod install && cd ../
```

3. Copy .env.example as .env.
4. Change API url in .env.

# Running the app

Make sure you have latest Xcode installed and simulator "iPhone 12" with iOS 16.2 or 16.4 is available.

1. Click on the "mlDemo.xcworkspace"
2. Open mlDemo
3. Click Products -> Product (from the menu bar) -> Clean Build Folder
4. Once clean build is done, click on Products -> Product (from the menu bar) -> Build

It should open "Metro Bundler" terminal and start building the app.
It should ask something like this:

```bash
To reload the app press "r"
To open developer menu press "d"
```

Wait for indexing and initializing datastore process to complete.

5. Enter "r" to reload the app.

Click on Podfile, and click Products -> Product (from the menu bar) -> Build

Again, click on Products inside Pods folder, and click Products -> Product (from the menu bar) -> Build

6. Go to your VS code terminal and open iOS folder and build the app.

```bash
cd ios
npm run ios
```

Usually it will open the simulator and run the app, but sometimes it doesn't.
if you recieve an error like this:

```bash
success Successfully built the app
...
...
error Failed to launch the app on simulator, An error was encountered processing the command (domain=com.apple.CoreSimulator.SimError, code=405):
Unable to lookup in current state: Shutdown
```

Can be done similar with android app.

# Deploying the TF Model (.h5) on GCP

1. Create a [GCP account](https://console.cloud.google.com/freetrial/signup/tos?_ga=2.25841725.1677013893.1627213171-706917375.1627193643&_gac=1.124122488.1627227734.Cj0KCQjwl_SHBhCQARIsAFIFRVVUZFV7wUg-DVxSlsnlIwSGWxib-owC-s9k6rjWVaF4y7kp1aUv5eQaAj2kEALw_wcB).
2. Create a [Project on GCP](https://cloud.google.com/appengine/docs/standard/nodejs/building-app/creating-project) (Keep note of the project id).
3. Create a [GCP bucket](https://console.cloud.google.com/storage/browser/).
4. Upload the tf .h5 model generate in the bucket in the path models/potato-model.h5.
5. Install Google Cloud SDK [Setup instructions](https://www.youtube.com/watch?v=OswWWJTRLIQ&list=PLPbgcxheSpE1gl5WkrwtmRiCwiGMM8NdH&index=7&ab_channel=codebasicsHindi).
6. Authenticate with Google Cloud SDK.

```bash
gcloud auth login
```

7. Run the deployment script inside gcp folder.

```bash
cd gcp
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id
```

8. Your model is now deployed.
9. Now, go to cloud functions and copy the Trigger URL.
10. Use Postman to test the GCF using the Trigger URL.

Inspiration: https://cloud.google.com/blog/products/ai-machine-learning/how-to-serve-deep-learning-models-using-tensorflow-2-0-with-cloud-functions

# To quantize the model

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
```
