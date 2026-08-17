from PIL import Image

from predict.predict_diagnoser import predict_diagnose
from predict.predict_gatekeeper import predict_validation


def predict_pipeline(image: Image.Image):

    gatekeeper_result = predict_validation(image)

    # Reject if model predicts Not Chest X-ray
    if gatekeeper_result["prediction"] == 0:
        return {
            "accepted": False,
            "message": "Uploaded image is not a chest X-ray.",
            "gatekeeper_confidence": gatekeeper_result["confidence"]
        }

    # Reject if confidence too low
    if gatekeeper_result["confidence"] < 0.85:
        return {
            "accepted": False,
            "message": "Gatekeeper is not sufficiently confident this is a chest X-ray.",
            "gatekeeper_confidence": gatekeeper_result["confidence"]
        }

    diagnosis_result = predict_diagnose(image)

    return {
        "accepted": True,
        "message": "Valid chest X-ray detected.",
        "gatekeeper_confidence": gatekeeper_result["confidence"],
        "prediction": diagnosis_result["prediction"],
        "confidence": diagnosis_result["confidence"],
        "probabilities": diagnosis_result["probabilities"],
    }