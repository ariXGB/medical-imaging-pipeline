from pathlib import Path
import shutil
import random

random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

def split_data(dirname:str):

    SOURCE_DIR = PROJECT_ROOT / "data" / "raw_data" / dirname
    DEST_DIR = PROJECT_ROOT / "data" / "dataset" / dirname

    for class_dir in SOURCE_DIR.iterdir():

        if not class_dir.is_dir():
            continue

        images = [f for f in class_dir.iterdir() if f.is_file()]

        random.shuffle(images)

        n = len(images)

        train_end = int(n * TRAIN_RATIO)
        val_end = train_end + int(n * VAL_RATIO)

        train_imgs = images[:train_end]
        val_imgs = images[train_end:val_end]
        test_imgs = images[val_end:]

        for split, split_imgs in {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs,
        }.items():

            target_dir = DEST_DIR / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for img in split_imgs:
                shutil.copy2(img, target_dir / img.name)

        print(
            f"{class_dir.name}: "
            f"train={len(train_imgs)}, "
            f"val={len(val_imgs)}, "
            f"test={len(test_imgs)}"
        )

    print("Dataset split complete.")

for dirname in ["diagnoser","gatekeeper"]:
    split_data(dirname=dirname)
