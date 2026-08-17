from torch.utils.data import DataLoader
from datasets_dataloaders.dataset import create_dataset

def create_dataloaders(model:str,isTraining=False):

    train_dataset,val_dataset,test_dataset = create_dataset(model)
    BATCH_SIZE = 32

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    if not isTraining:
        return test_loader

    return train_loader,val_loader