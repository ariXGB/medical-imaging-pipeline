import streamlit as st

from PIL import Image

import os
from dotenv import load_dotenv

import requests

load_dotenv()

FAST_API_URL = os.getenv("FAST_API_URL")

st.set_page_config(
    page_title="Chest X-Ray AI Diagnoser",
    page_icon="🩻",
    layout="centered"
)

st.title("🩻 Chest X-Ray AI Diagnoser")

st.markdown(
    """
    Upload a medical image.

    The image is first validated by the Gatekeeper model.
    If it is identified as a chest X-ray, it is passed to the
    Diagnoser model for classification.
    """
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze"):

        with st.spinner("Running AI pipeline..."):

            try:

                response = requests.post(
                    url=f"{FAST_API_URL}/predict/",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

            except requests.exceptions.SSLError as e:

                st.error(
                    f"SSL Certificate Error:\n{e}"
                )

                st.stop()

            except requests.exceptions.ConnectionError as e:

                st.error(
                    f"Unable to connect to FastAPI:\n{e}"
                )

                st.stop()

            except requests.exceptions.Timeout:

                st.error(
                    "Request timed out."
                )

                st.stop()

            except requests.exceptions.RequestException as e:

                st.error(
                    f"API Error:\n{e}"
                )

                st.stop()

        
        # Gatekeeper Rejected
        

        if not result["accepted"]:

            st.error("❌ Image Rejected")

            st.write(
                result["message"]
            )

            st.metric(
                "Gatekeeper Confidence",
                f"{result['gatekeeper_confidence']*100:.2f}%"
            )

       
        # Gatekeeper Accepted
        

        else:

            st.success(
                "✅ Valid Chest X-Ray Detected"
            )

            st.metric(
                "Gatekeeper Confidence",
                f"{result['gatekeeper_confidence']*100:.2f}%"
            )

            st.divider()

            st.subheader("Diagnosis")

            st.metric(
                "Prediction",
                result["prediction"]
            )

            st.metric(
                "Confidence",
                f"{result['confidence']*100:.2f}%"
            )

            st.divider()

            st.subheader(
                "Class Probabilities"
            )

            for cls, prob in result[
                "probabilities"
            ].items():

                st.write(
                    f"{cls}: {prob*100:.2f}%"
                )

                st.progress(
                    float(prob)
                )