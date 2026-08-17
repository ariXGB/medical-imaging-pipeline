from lightning import Trainer
import torch

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from model_configs.diagnoser import ChestClassifier
print("Diagnoser Model Imported")
from datasets_dataloaders.dataloader import create_dataloaders

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device - {device}")

def main():
    model_type = "diagnoser"
    train_loader,val_loader = create_dataloaders(model=model_type,isTraining=True)
    num_classes = 3

    model = ChestClassifier(
        num_classes=num_classes,
        freeze_backbone=True,
        lr=1e-3,
        weight_decay=1e-4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="models/diagnoser",
        filename="best",
        monitor="val_acc",
        mode="max",
        save_top_k=1
    )

    logger = TensorBoardLogger(
    save_dir="lightning_logs",
    name="diagnoser"
)

    trainer = Trainer(
        logger=logger,
        max_epochs=70,
        accelerator="gpu",
        devices=1,
        precision='32',
        callbacks= [checkpoint_callback],
        log_every_n_steps=5
    )
    print("Trainable:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    print("Training Diagnoser Model-:")
    trainer.fit(
        model,
        train_loader,
        val_loader
    )

if __name__ == '__main__':
    main()