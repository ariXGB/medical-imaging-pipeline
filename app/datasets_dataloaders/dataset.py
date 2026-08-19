from pathlib import Path
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_SIZE = 224

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
val_test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def create_dataset(model:str):
    DATASET_DIR = PROJECT_ROOT / "data" / "dataset" / f"{model}"

    train_dataset = datasets.ImageFolder(root=DATASET_DIR / "train",transform=train_transforms)
    val_dataset = datasets.ImageFolder( root=DATASET_DIR / "val",transform=val_test_transforms)
    test_dataset = datasets.ImageFolder(root=DATASET_DIR / "test",transform=val_test_transforms)

    return (train_dataset,val_dataset,test_dataset)