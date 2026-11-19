# Annotation Guidelines — suppilovahvero

Status: Draft — pending pilot annotation.

## Target and class ID

The target class is suppilovahvero (Craterellus tubaeformis).
The YOLO class ID is 0.

Annotate only target mushrooms. Other mushroom species and
background objects do not receive boxes.

## Bounding boxes

Draw one tight bounding box per identifiable target mushroom.
Include each mushroom separately when individuals can be distinguished
in a cluster.

Rules for partially hidden, truncated and uncertain objects
must be clarified before full annotation.

## Export

Export YOLO detection TXT labels to dataset/consolidated/labels/.
Each label file must match its image filename stem.

Each object is represented by one line:

class_id x_center y_center width height

All bounding box coordinates are normalized to [0, 1].
Normalize x_center and width by image width,
and y_center and height by image height.

## Negative and unresolved images

After visually confirming that an image contains no target,
provide an empty TXT label file.

Missing or empty labels alone do not prove that an image is negative.
Record uncertain cases for resolution before final delivery.

## CNN labels versus YOLO class IDs

CNN image-level labels are:
- Target_Absent = 0
- Target_Present = 1

An image is positive if it contains at least one target mushroom,
even when other objects or mushroom species are also present.

YOLO class ID 0 identifies the target category.
It does not mean that the target is absent.
Do not write CNN image-level labels into YOLO detection TXT files.

## Source folders versus target presence

The source folders mushroom/ and no_mushroom/ distinguish
images containing any mushroom from images containing none.
They do not directly define target presence.

Review every image and annotate only suppilovahvero.
An image containing only other mushroom species is target-negative.

Final CNN labels must reflect verified target presence,
not source folder names. Missing or empty annotation files
must not be treated as verified negatives without a completed review.

## Shared dataset split

Eddie and Sasha manage the shared train/validation/test split,
approximately 70%/15%/15%, before augmentation.

Both CNN classification and YOLO detection must use exactly
the same image membership in each split. Near-duplicates and
images from the same short photo session must remain together.

Annotation filenames must preserve their matching image stems
so labels can follow the assigned split.

Positive and negative image counts per split will be documented
once the shared split is available.

## Quality review

Visually review all images and boxes before final delivery.
Validate the exported label files and record corrections
and unresolved cases.

---

## Annotation Completion and Dataset Handoff - 19 November 2026

### Annotation results

Manual annotation of the cleaned dataset is complete.

- Total images reviewed and verified: 399
- Target-present images: 349
- Target-absent images: 50
- Total YOLO bounding boxes: 1,533
- Annotation batches completed: 9
- Target species: Craterellus tubaeformis (suppilovahvero)
- YOLO class ID: 0

Each target-present image has a YOLO TXT file containing its bounding boxes.
Each target-absent image has an empty YOLO TXT file.

### Final annotation files

- `dataset/consolidated/labels/`: 399 YOLO TXT files.
- `dataset/consolidated/cnn_labels.csv`: 399 image-level classification labels.
- `dataset/label_status.csv`: annotation review status and box counts.

CNN classification labels:
- 1 = Target_Present (349 images)
- 0 = Target_Absent (50 images)

CNN labels and YOLO class IDs have different meanings.

### Dataset validation

The following checks were completed:

- All 399 images have matching YOLO TXT files.
- All 399 CNN labels correspond to existing images.
- All 349 positive images have non-empty YOLO TXT files.
- All 50 negative images have empty YOLO TXT files.
- No CNN/YOLO label inconsistencies were detected.
- All 798 consolidated JPG/TXT files matched their sources by SHA-256.
- No exact duplicate JPG files were detected by SHA-256.
- YOLO annotation syntax and coordinate checks passed.

Two visually similar image pairs were identified during preliminary
near-duplicate analysis. Related images should remain in the same
training, validation or test partition to reduce data leakage.

### Image storage

The 399 cleaned JPG images are stored in the team's Cloudflare R2 bucket:

- `craterellus/clean/photos/mushroom/`
- `craterellus/clean/photos/no_mushroom/`

All 399 local JPG files matched the corresponding R2 files by
filename and file size. This was not a full content-hash comparison.

### Dataset splitting

The final train/validation/test split has not yet been performed.

Eddie plans to implement splitting based on GPS coordinates and
capture timestamps. CNN and YOLO should use identical image
membership in each split.

The annotation files are ready for team handoff. The final dataset
structure and split generation remain separate tasks.


### How to use the annotated dataset

1. Retrieve the 399 cleaned JPG images from Cloudflare R2 under
   `craterellus/clean/photos/mushroom/` and
   `craterellus/clean/photos/no_mushroom/`.

2. Match each JPG with its YOLO TXT file in
   `dataset/consolidated/labels/` using the same filename stem.
   Empty TXT files are intentional for target-absent images.

3. For CNN classification, use
   `dataset/consolidated/cnn_labels.csv`.
   The CSV columns are `filename` and `label`, where 1 means
   Target_Present and 0 means Target_Absent.

4. Create train/validation/test partitions before model training.
   Use GPS coordinates and capture timestamps to group related
   images and reduce data leakage. Keep visually similar images
   together where appropriate.

5. Use identical image membership in each partition for CNN and YOLO.
   Generate model-specific folder structures and YOLO configuration
   files from the final split.

6. Record the final split counts and any exclusions. The split has
   not yet been generated as part of this annotation handoff.

The original JPG images are stored in R2 and are not included
in the Git repository.
