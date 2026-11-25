"""
Building the YOLO and CNN folders from split.csv, the clean photos and labels

"""

import shutil
from pathlib import Path

import pandas as pd

from nice_printing import section


# Which photo goes where

LABELS = "dataset/consolidated/labels"            # the YOLO .txt files
YOLO = Path("dataset/yolo")
CNN = Path("dataset/cnn")

split = pd.read_csv("dataset/split.csv")
catalog = pd.read_csv("dataset/clean/catalog.csv")
photos = split.merge(catalog[["filename", "path", "has_mushroom"]], on="filename")

section("1. Photos per split")
print("Total:        ", len(photos))
print("Train:        ", (photos["split"] == "train").sum())
print("Val:          ", (photos["split"] == "val").sum())
print("Test:         ", (photos["split"] == "test").sum())


# Making folders and copying them

# Making sure we start clean every run

shutil.rmtree(YOLO, ignore_errors=True)
shutil.rmtree(CNN, ignore_errors=True)

for part in ["train", "val", "test"]:
    (YOLO / "images" / part).mkdir(parents=True)
    (YOLO / "labels" / part).mkdir(parents=True)
    (CNN / part / "Target_Present").mkdir(parents=True)
    (CNN / part / "Target_Absent").mkdir(parents=True)

for photo in photos.to_dict("records"):
    image = Path("dataset/clean") / photo["path"]
    label = Path(LABELS) / photo["filename"].replace(".jpg", ".txt")
    part = photo["split"]

    if photo["has_mushroom"] == 1:
        target = "Target_Present"
    else:
        target = "Target_Absent"

    shutil.copy(image, YOLO / "images" / part)
    shutil.copy(label, YOLO / "labels" / part)
    shutil.copy(image, CNN / part / target)

section("2. Folders built")
print("dataset/yolo/ and dataset/cnn/")


# data.yaml for YOLO is below: where the images are and what class 0 is called
# (YOLO finds the labels itself in labels/ next to images/)

yaml_text = f"""path: {YOLO.resolve()}
train: images/train
val: images/val
test: images/test
names:
  0: suppilovahvero
"""
(YOLO / "data.yaml").write_text(yaml_text)

section("3. data.yaml")
print(yaml_text)
