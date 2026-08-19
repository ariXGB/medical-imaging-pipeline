from datasets_dataloaders.dataloader import create_dataloaders
from model_configs.diagnoser import ChestClassifier
from utils.log_metrics import log_metrics
import torch
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassRecall
)

checkpoint = "models/diagnoser/best.ckpt"


def test():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ChestClassifier.load_from_checkpoint(checkpoint)

    model.to(device)
    model.eval()

    test_loader = create_dataloaders(model="diagnoser",isTraining=False)

    acc = MulticlassAccuracy(num_classes=3).to(device)
    f1 = MulticlassF1Score(num_classes=3,average="macro").to(device)
    recall = MulticlassRecall(num_classes=3,average="macro").to(device)

    with torch.no_grad():

        for x, y in test_loader:

            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            preds = torch.argmax(logits,dim=1)

            acc.update(preds, y)
            f1.update(preds, y)
            recall.update(preds, y)

    metrics = {
        "model_type" : "diagnoser",
        "accuracy" : f"{acc.compute():.2f}%",
        "F1" : f"{f1.compute():.2f}%",
        "Recall" : f"{recall.compute():.2f}%"
    }

    print(
        f"Accuracy : {metrics['accuracy']}"
    )

    print(
        f"F1 Score : {metrics['F1']}"
    )

    print(
        f"Recall    : {metrics['Recall']}"
    )

    log_metrics(metrics=metrics,model_type="diagnoser")

if __name__ == "__main__":
    test()