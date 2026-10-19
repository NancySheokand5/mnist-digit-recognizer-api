MNIST Digit Recognizer API

A deep learning project that recognizes handwritten digits (0-9) from an image.

A Convolutional Neural Network (CNN) is trained on the MNIST dataset with TensorFlow/Keras, served through a FastAPI REST API, and packaged in a Docker container so it runs the same way on any computer.

Problem

Given a picture of one handwritten digit, predict which digit (0-9) it is, and return how confident the model is.

Dataset

MNIST is a built-in Keras dataset.

Split	Images	Notes
Train	54,000	used to teach the model
Validation	6,000	used to check progress during training
Test	10,000	used once for the final score

Each image is 28 x 28 pixels, grayscale, with a white digit on a black background.

Model Architecture
Input (28, 28, 1)
  -> Conv2D (32 filters, 3x3, ReLU)
  -> MaxPooling2D (2x2)
  -> Conv2D (64 filters, 3x3, ReLU)
  -> MaxPooling2D (2x2)
  -> Flatten
  -> Dropout (0.5)
  -> Dense (128, ReLU)
  -> Dense (10, Softmax)
Total parameters: 225,034
Optimizer: Adam
Loss: sparse categorical crossentropy
Epochs: 5, batch size: 128
Results
Metric	Score
Test accuracy	98.89%
Validation accuracy (last epoch)	98.85%
How the API handles real photos

MNIST digits are cropped, centered, and about 20 x 20 pixels inside a 28 x 28 box. A simple resize makes real photos look very different from the training data, so the API cleans every image first:

Fix photo rotation and convert to grayscale
Stretch the contrast
Detect the background color and invert it if needed (digit becomes white on black)
Remove faint noise
Crop to the digit
Resize so the longest side is 20 pixels (keeps the shape)
Paste on a black 28 x 28 canvas and center it by its center of mass

The cleaned 28 x 28 image is returned in the response so wrong predictions are easy to debug.

Project Structure
.
├── app.py               # FastAPI backend (model loading, preprocessing, /predict)
├── frontend.py          # Streamlit web page to upload an image
├── train_model.ipynb    # Notebook: load data, train, evaluate, save the model
├── mnist_cnn.keras      # Trained model
├── requirements.txt     # Python libraries for the API
├── Dockerfile           # Builds the API container
├── .dockerignore
├── .gitignore
└── screenshots/         # Postman and Docker screenshots
Run with Docker (recommended)

You only need Docker Desktop.

1. Build the image (the first build takes several minutes because TensorFlow is large):

bash
docker build -t mnist-api .

2. Run the container:

bash
docker run -p 8000:8000 --name mnist-container mnist-api

3. Open the API docs: http://127.0.0.1:8000/docs

Stop and restart later:

bash
docker stop mnist-container
docker start mnist-container
Run without Docker
bash
python -m pip install -r requirements.txt
python -m uvicorn app:app
Run the frontend (optional)

With the API running, open a second terminal:

bash
python -m pip install streamlit requests pandas
python -m streamlit run frontend.py

Open http://localhost:8501, upload an image, and click Predict.
<img width="1540" height="467" alt="image" src="https://github.com/user-attachments/assets/8d3474a8-f8a3-4762-b0e7-3ed75b0b6228" />

API Usage
GET /

Checks that the API is running.

POST /predict

Send an image as multipart/form-data with the field name file (PNG or JPG, max 5 MB).

Example (PowerShell or Command Prompt):

bash
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@mnist.jpg"

Example response:

json
{
  "digit": 7,
  "confidence": 99.87,
  "probabilities": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9987, 0.0, 0.0013],
  "processed_image": "iVBORw0KGgo..."
}
Field	Meaning
digit	The predicted digit (0-9)
confidence	Probability of the predicted digit, in percent
probabilities	Probability of each digit 0-9
processed_image	The 28 x 28 image the model saw, as base64 PNG

Errors

Status	Reason
400	File is not a PNG/JPG, is larger than 5 MB, is not a valid image, or contains no digit
422	The file field is missing
Screenshots

Postman test (200 OK)

<img width="1456" height="819" alt="image" src="https://github.com/user-attachments/assets/5467c956-e7e8-4217-bb12-e7caa91a98ab" />


Docker container running

<img width="1615" height="920" alt="image" src="https://github.com/user-attachments/assets/3796e3a2-173f-47e3-8dba-f0f40c8f7b24" />


Tech Stack
Model: Python, TensorFlow, Keras, NumPy
Image processing: Pillow
Backend: FastAPI, Uvicorn, Pydantic
Frontend: Streamlit
Testing: Postman
Deployment: Docker
What I Learned
Loading, normalizing, and reshaping image data for a CNN
Building, training, evaluating, and saving a Keras model
Why preprocessing at prediction time must match preprocessing at training time
Building a REST API with FastAPI and testing it with Postman
Packaging a machine learning app in a Docker container
Future Improvements
Add data augmentation (rotation, shift) to handle messy handwriting better
Add a drawing canvas to the frontend
Add automated tests for the API
Deploy the container to a cloud service
Author

Nancy Sheokand GitHub: @NancySheokand5
