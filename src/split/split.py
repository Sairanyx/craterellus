"""
Splits the photos into train / val / test by estimated patches of mushrooms,
so we are trying to limit the chnaces of the same patch ending up in two parts

"""


import pandas as pd
import math
from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedGroupKFold


def section(title):
    print()
    print(title)
    print("-" * 40)


# Loading, sorting reindexing and printing to check

catalog = pd.read_csv("dataset/clean/catalog.csv")
photos = catalog[(catalog["type"] == "photo") & (catalog["source"] != "drone")]      # only photos: videos are for the demo, drone photos are test samples
photos = photos.sort_values("time")                 # all photos in the order they were taken
photos = photos.reset_index(drop=True)              # numbering the rows from 0 as now they have the old index

section("1. Photos")
print("Total:        ", len(photos))
print("With mushroom:", (photos["has_mushroom"] == 1).sum())
print("Empty:        ", (photos["has_mushroom"] == 0).sum())


# GPS calculations to make group the mushroom photos per patch of 3 m radius

METRES_PER_DEGREE = 111_320
cos_lat = math.cos(math.radians(60.48))             # Turku is at 60.48 degrees north

photos["y"] = photos["lat"] * METRES_PER_DEGREE     # meters north
photos["x"] = photos["lon"] * METRES_PER_DEGREE * cos_lat    # meters east


# Finding the patches

PATCH_RADIUS = 3                                    # photos closer than 3m are part of the same patch

photos["patch"] = None

for label, name in [(1, "mushroom"), (0, "empty")]:
    group = photos[(photos["has_mushroom"] == label) & photos["lat"].notna()]
    numbers = DBSCAN(eps=PATCH_RADIUS, min_samples=1).fit_predict(group[["x", "y"]])
    photos.loc[group.index, "patch"] = [f"{name}_patch_{n + 1}" for n in numbers]

section("2. Patches (photos within 3 m)")
print("Patches:      ", photos["patch"].nunique())
print("Biggest 5:    ", photos["patch"].value_counts().head(5).to_dict())


# Using the 4 photos without GPS

photos["patch"] = photos.groupby("source")["patch"].ffill()

print("No patch:     ", photos["patch"].isna().sum(), "(should be 0)")  # just a check and should be 


# Sorting the patches

cut = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=0)

slices = []

for _, rows in cut.split(photos, photos["has_mushroom"], photos["patch"]):
    slices.append(rows)

slices = sorted(slices, key=len)                                        # taking the smallest slice first                              

photos["split"] = "train"
photos.loc[slices[0], "split"] = "test"
photos.loc[slices[1], "split"] = "val"

section("3. Split")
table = photos.groupby(["split", "has_mushroom"]).size().unstack()
print(table.loc[["train", "val", "test"]])


# Saving result to csv

photos[["filename", "split", "patch"]].to_csv("dataset/split.csv", index=False)

section("4. Saved")
print("dataset/split.csv:", len(photos), "photos")