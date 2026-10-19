import io
import base64

import requests
import pandas as pd
import streamlit as st
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="Digit Recognizer", page_icon="🔢")
st.title("🔢 Handwritten Digit Recognizer")
st.write("Upload a picture of one handwritten digit (0-9) and the CNN will guess it.")

uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    st.image(uploaded, caption="Your image", width=200)

    if st.button("Predict"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach the backend. Start it first with: python -m uvicorn app:app")
                st.stop()

        if response.status_code == 200:
            data = response.json()

            st.success(f"Predicted digit: {data['digit']}")
            st.metric("Confidence", f"{data['confidence']} %")

            # Show what the model really saw
            processed = Image.open(io.BytesIO(base64.b64decode(data["processed_image"])))
            st.image(processed.resize((140, 140), Image.NEAREST), caption="What the model saw (28x28)")

            # Chart of all 10 probabilities
            st.subheader("Probability of each digit")
            st.bar_chart(pd.DataFrame({"probability": data["probabilities"]}))
        else:
            st.error(response.json().get("detail", "Something went wrong"))