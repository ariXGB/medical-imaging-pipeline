from lightning import Trainer
from datasets_dataloaders.dataloader import create_dataloaders
from model_configs.gatekeeper import Gatekeeper
from utils.log_metrics import log_metrics
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall
)

checkpoint = "models/gatekeeper/best.ckpt"


def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Gatekeeper.load_from_checkpoint(checkpoint)
    test_loader = create_dataloaders(model="gatekeeper",isTraining=False)

    trainer = Trainer(
        accelerator="gpu",
        devices=1
    )

    metrics = trainer.test(model=model,dataloaders=test_loader)[0]

    for k,v in metrics.items():
        print(f"{k} : {v}")

    log_metrics(metrics=metrics,model_type="gatekeeper")



if __name__ == "__main__":
    test()