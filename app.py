import io
import base64

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow import keras

app = FastAPI(title="MNIST Digit Recognizer")

# Load the model once when the server starts
model = keras.models.load_model("mnist_cnn.keras")

MAX_FILE_SIZE = 5 * 1024 * 1024   # 5 MB


# This describes what the API sends back
class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    probabilities: list[float]
    processed_image: str          # the 28x28 image the model saw (base64 text)


def preprocess(img):
    """Turn any photo of a digit into a clean MNIST-style 28x28 image."""

    # 1. Fix phone-photo rotation, then make it gray
    img = ImageOps.exif_transpose(img).convert("L")

    # 2. Stretch contrast so dark is really dark and light is really light
    img = ImageOps.autocontrast(img)
    arr = np.array(img).astype("float32")

    # 3. MNIST = white digit on black. Look at the edges of the picture:
    #    if the background is bright, flip the colors.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if np.median(border) > 127:
        arr = 255 - arr

    # 4. Remove faint noise
    arr[arr < 50] = 0

    # 5. Find where the digit is and crop around it
    ys, xs = np.where(arr > 0)
    if len(ys) == 0:
        raise ValueError("No digit found in the image")
    arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # 6. Resize so the longest side is 20 pixels (keep the shape)
    h, w = arr.shape
    scale = 20.0 / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = Image.fromarray(arr.astype("uint8")).resize((new_w, new_h), Image.LANCZOS)

    # 7. Paste it in the middle of a black 28x28 canvas
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(digit, ((28 - new_w) // 2, (28 - new_h) // 2))

    # 8. Move the digit so its "center of mass" is exactly in the center (like MNIST)
    a = np.array(canvas).astype("float32")
    total = a.sum()
    center_y = (a * np.arange(28)[:, None]).sum() / total
    center_x = (a * np.arange(28)[None, :]).sum() / total
    a = np.roll(a, (int(round(14 - center_y)), int(round(14 - center_x))), axis=(0, 1))

    return Image.fromarray(a.astype("uint8"))


@app.get("/")
def home():
    return {"message": "Digit recognizer API is running. Go to /docs to try it."}


@app.post("/predict", response_model=PredictionResponse)
def predict(file: UploadFile = File(...)):
    # Check the file type
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(status_code=400, detail="Please upload a PNG or JPG image")

    # Read the file and check its size
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File is too big (max 5 MB)")

    # Open it as an image
    try:
        img = Image.open(io.BytesIO(contents))
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="This is not a valid image file")

    # Clean the image
    try:
        clean_img = preprocess(img)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))

    # Predict
    img_array = np.array(clean_img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    probs = model(img_array, training=False).numpy()[0]

    # Turn the clean image into text so it can be sent as JSON
    buffer = io.BytesIO()
    clean_img.save(buffer, format="PNG")
    image_text = base64.b64encode(buffer.getvalue()).decode()

    return PredictionResponse(
        digit=int(np.argmax(probs)),
        confidence=round(float(np.max(probs)) * 100, 2),
        probabilities=[round(float(p), 4) for p in probs],
        processed_image=image_text,
    )

