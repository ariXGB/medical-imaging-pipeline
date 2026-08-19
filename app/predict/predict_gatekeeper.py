from PIL import Image
from lightning import Trainer
import torch
from torch.utils.data import DataLoader, TensorDataset
from model_configs.gatekeeper import Gatekeeper
from datasets_dataloaders.dataset import val_test_transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = val_test_transforms
MODEL = Gatekeeper.load_from_checkpoint("models/gatekeeper/best.ckpt")
MODEL.to(DEVICE)

def predict_validation(image: Image.Image):

    image = image.convert("RGB")
    tensor = TRANSFORM(image)
    tensor = tensor.unsqueeze(0) # type: ignore
    tensor = tensor.to(DEVICE)

    trainer = Trainer(
        accelerator="gpu",
        devices=1
    )
    predict_loader = DataLoader(TensorDataset(tensor),batch_size=1)
    res = trainer.predict(model=MODEL,dataloaders=predict_loader)[0] # type: ignore

    return res