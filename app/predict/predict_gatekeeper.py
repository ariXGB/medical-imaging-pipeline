from PIL import Image

import torch
import torch.nn.functional as F

from torchvision import transforms

from model_configs.gatekeeper import Gatekeeper
from datasets_dataloaders.dataset import val_test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Chest_X_ray","Not_Chest_x_ray"]

TRANSFORM = val_test_transforms

MODEL = Gatekeeper.load_from_checkpoint("models/gatekeeper/best.ckpt")

MODEL.to(DEVICE)
MODEL.eval()

def predict_validation(image: Image.Image):

    image = image.convert("RGB")

    tensor = TRANSFORM(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        logits = MODEL(tensor)

        probs = F.softmax(logits,dim=1)

        confidence, pred = torch.max(probs,dim=1)

    return {
        "prediction": CLASS_NAMES[pred.item()],
        "confidence": float(confidence.item()),
        }