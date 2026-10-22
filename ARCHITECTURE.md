# Architecture — Suppilovahvero Detector

**Goal:** detect *Craterellus tubaeformis* (suppilovahvero / funnel chanterelle) in forest images.
1. **Custom CNN** answers *is there a mushroom?* (binary, Milestone 2)
2. **Fine-tuned YOLOv5** answers *where is it?* (bounding boxes, Milestone 3)
3. **Live demo:** a Pi camera on a drone streams over a terrarium, and the laptop draws the boxes in real time.

---

## 1. System overview

```
 ┌──────────────── DATA (M1) ────────────────┐
 │ 5× phones ──► raw_images/ ──► preprocess.py│
 │ (HEIC→JPG, EXIF rotate, ≤1920px)           │
 │            ──► images/ ──► CVAT ──► labels/ │  (YOLO .txt + Pascal VOC .xml)
 └────────────────────┬───────────────────────┘
                      │ split.py (by location, 70/15/15)
          ┌───────────┴────────────┐
          ▼                        ▼
 ┌─── CNN (M2) ───┐       ┌─── YOLO (M3) ───┐
 │ Keras, custom  │       │ Ultralytics     │
 │ binary, 224px  │       │ YOLOv5s → 5n    │
 │ → cnn_best.h5  │       │ → yolo_best.pt  │
 └───────┬────────┘       └────────┬────────┘
         └──────────┬──────────────┘
                    ▼
 ┌──────────── DEMO / PANEL ─────────────┐
 │ Gradio app (laptop)                    │
 │  Tab 1: upload image → CNN + YOLO      │
 │  Tab 2: GoPro video → tracked + count  │
 │  Tab 3: live drone stream → boxes      │
 │  Tab 4: metrics (training curves, P/R) │
 └────────────────▲───────────────────────┘
                  │ Wi-Fi hotspot (RTSP/MJPEG, 640×480)
 ┌────────────────┴───────┐
 │ Drone + Pi Zero 2 W +  │  ← flies over terrarium (indoor, no licence)
 │ Pi Camera (stream only)│
 └────────────────────────┘
```

---

## 2. Stack

| Layer | Tool | Why |
|---|---|---|
| Language | Python 3.11 | everything in one language |
| Preprocessing | Pillow, pillow-heif | fix HEIC + EXIF rotation |
| Annotation | **CVAT** (cloud, shared project) | 5 people annotate in parallel, exports YOLO + VOC |
| CNN | **TensorFlow / Keras** | simple API, saves to `.h5` |
| Detection | **Ultralytics YOLOv5** | required by the spec |
| Tracking | ByteTrack (built into Ultralytics) | counts unique mushrooms in video |
| Explainability | Grad-CAM (tf-keras-vis) | proves the CNN looks at the mushroom |
| Compute | CSC (Puhti/Mahti) GPU or Colab | training |
| Experiment log | CSV + matplotlib (optionally W&B) | tuning-journey table |
| Panel | **Gradio** | demo UI in about 100 lines |
| Edge | Raspberry Pi Zero 2 W + Camera Module, `rpicam-vid` / MediaMTX | low-latency stream |
| Code | GitHub repo, one branch per milestone | division of tasks visible in commits |
| Storage | Shared drive (originals) + repo (code) + GitHub Release (weights) | simple, enough |

### Not used (deliberately)
- **S3 / PostgreSQL:** ~300 images and one metadata CSV don't need them. They add setup and failure points, and earn **0 grading points**. If we want "data engineering" polish, use **SQLite** (one file, zero setup) for the detection log behind the panel's stats tab.

---

## 3. Data (Milestone 1)

**Target:** ≥250 positive + ≥30 negative (~10%) unique images.

| Who | Positives | Negatives | Prefix |
|---|---|---|---|
| each of 5 members | ~50 | ~6 | `ED_`, `XX_`, … |

**Shot list per patch:** 1.5–2 m @45° · ~1 m @60–90° · 0.5–1 m top-down · ground level · partly hidden.
**Negatives:** same forest floor without mushrooms, yellow leaves, pine cones, lookalike mushrooms.
**Rules:** JPG ("Most Compatible" on iPhone), no filters or burst, avoid near-duplicates, keep originals.

**metadata.csv** (one row per image):
```
filename, author, location_id, date, height_m, angle_deg, has_mushroom, n_instances, split
```

**Split by `location_id`** so the same patch never appears in both train and test (prevents leakage).

**Deliverable structure (`dataset.zip`):**
```
images/        *.jpg
annotations/   *.xml   (Pascal VOC)
labels/        *.txt   (YOLO: class cx cy w h, normalized)
metadata.csv
```

---

## 4. CNN (Milestone 2)

- **Input:** 224×224×3, rescale 1/255
- **Baseline:** 4× [Conv2D(3×3) → ReLU → MaxPool] (32-64-128-128) → Flatten → Dense(128) → Dense(1, sigmoid). No regularization, fixed 30 epochs.
- **Diagnosis:** compare train vs val accuracy → high bias or high variance.
- **Fix (likely high variance, small dataset):** Dropout(0.3–0.5) + L2(1e-4) on dense layers, augmentation (flip, rotate, brightness), GlobalAveragePooling instead of Flatten.
- **Callbacks:** `EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)`, `ModelCheckpoint("cnn_best.h5", save_best_only=True)`.
- **Class imbalance (~90/10):** `class_weight` and report **precision/recall**, not only accuracy.
- **Outputs:** tuning-journey table (baseline → each change → final), test Accuracy / Precision / Recall / confusion matrix, Grad-CAM samples.

---

## 5. YOLO (Milestone 3)

- **Model:** `yolov5s.pt` pretrained on COCO → fine-tune. Export `yolov5n` for live demo speed.
- **Config `data.yaml`:**
  ```yaml
  path: dataset
  train: images/train
  val: images/val
  test: images/test
  names: {0: suppilovahvero}
  ```
- **Train:** `imgsz=640, epochs=100, batch=16, patience=20`, default augmentations (mosaic, HSV).
- **Evaluate:** mAP@0.5, mAP@0.5:0.95, P/R on the test set, plus saved prediction images.
- **Extras:**
  - SAHI (tiled inference for tiny/far mushrooms): compare with vs. without
  - ByteTrack on GoPro video → "N unique mushrooms found"

---

## 6. Demo / Edge

| Stage | Input | Shows |
|---|---|---|
| 1 | test images | CNN verdict + YOLO boxes + Grad-CAM |
| 2 | GoPro "drone-sim" forest video (**test only, never trained on**) | boxes + tracking + count |
| 3 | live drone over terrarium | real-time boxes on projector |

**Live pipeline:** Pi streams 640×480 H.264 → hotspot → laptop `cv2.VideoCapture` (buffer = 1, newest frame only) → YOLOv5n (`imgsz=416`) → Gradio. Target latency **< 0.5 s**.

**Terrarium:** moss, yellow leaves, pine cones, real suppilovahvero + lookalike negatives.

**Fallbacks:** pre-recorded perfect flight · Pi camera on a stick if the drone can't lift it.

---

## 7. Repo layout

```
mushroom-detector/
├── data/            (gitignored; shared drive)
├── src/
│   ├── preprocess.py
│   ├── split.py
│   ├── voc_to_yolo.py
│   ├── cnn_train.ipynb
│   ├── yolo_train.ipynb
│   ├── gradcam.py
│   └── stream_infer.py
├── app/panel.py     (Gradio)
├── configs/data.yaml
├── models/          (cnn_best.h5, yolo_best.pt — via Release)
├── results/         (plots, tables, prediction images, videos)
├── report/
└── ARCHITECTURE.md
```

---

## 8. Roles

| Role | Owns |
|---|---|
| Project Manager | timeline, report, data split + metadata |
| Data lead | collection protocol, preprocessing, CVAT |
| CNN lead | M2 notebook, tuning table, Grad-CAM |
| YOLO lead | M3 training, SAHI, tracking |
| Demo/Edge lead | Pi stream, drone, terrarium, Gradio panel |

Everyone collects ~56 images and annotates their own.

---

## 9. Timeline

| When | Milestone |
|---|---|
| 25–31 Oct | forest collection (before frost) + GoPro footage |
| 1–7 Nov | preprocessing, CVAT annotation, split → **M1 zip** |
| 8–21 Nov | CNN baseline → diagnosis → optimization → **M2** |
| 15–28 Nov | YOLO fine-tuning + extras → **M3** |
| 22 Nov–2 Dec | Pi stream, terrarium, Gradio panel, rehearsal |
| Week 49 | presentation + report |

---

## 10. Risks

| Risk | Mitigation |
|---|---|
| Frost / too few mushrooms | collect early; backup object we own |
| Data leakage | split by location; GoPro video never in training |
| Drone can't lift Pi | handheld "stick drone" |
| Live demo fails | pre-recorded video |
| Class imbalance | class weights + precision/recall |
