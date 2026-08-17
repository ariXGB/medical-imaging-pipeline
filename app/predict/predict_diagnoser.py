from PIL import Image

import torch

from model_configs.diagnoser import ChestClassifier
from datasets_dataloaders.dataset import val_test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Normal","Pneumonia","Tuberculosis"]

TRANSFORM = val_test_transforms

MODEL = ChestClassifier.load_from_checkpoint("models/diagnoser/best.ckpt")

MODEL.to(DEVICE)
MODEL.eval()



def predict_diagnose(image: Image.Image):

    image = image.convert("RGB")

    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits = MODEL(tensor)

        probs = torch.softmax(logits,dim=1)

        confidence, pred = torch.max(probs,dim=1)


    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": float(confidence.item()),
        "probabilities": {
            cls: float(prob)
            for cls, prob in zip(
                CLASS_NAMES,
                probs[0]
            )
        }
    }