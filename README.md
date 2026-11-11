# Craterellus 🍄

A drone that finds **suppilovahvero** (*Craterellus tubaeformis*) on the forest floor, without trampling it.

**What we build:**
- **CNN:** says whether a photo has a mushroom.
- **YOLO:** says whether there's a mushroom, and draws a box around each one.
- **Demo:** a Gradio app runs both models on photos and video, and YOLO runs live on a Ryze Tello flying over a 1 m × 1 m terrarium.

Who does what and some ideas on what to use: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

```bash
git clone https://github.com/Sairanyx/craterellus.git
cd craterellus
uv sync                    # install packages (Python 3.12)
cp .env.example .env       # then paste in the R2 keys (sent privately)
```
- **New package:** `uv add <name>`, then commit `pyproject.toml` and `uv.lock`.

## Data

Photos, videos and trained models are in **Cloudflare R2**, not in git.
**R2 object storage** is Cloudflare's online storage for big files: a shared drive that scripts can read from and write to with one command.

### The `craterellus` bucket
```
craterellus/
├── raw/                    originals as taken (locked: can't be changed or deleted)
│   ├── photos_eddie/
│   ├── photos_benjamin/
│   ├── photos_drone/
│   ├── videos_drone/
│   ├── videos_mushrooms/
│   └── videos_no_mushrooms/
├── clean/                  pipeline output, what everyone works from
│   ├── photos/
│   │   ├── mushroom/
│   │   └── no_mushroom/
│   ├── videos/
│   │   ├── mushroom/
│   │   └── no_mushroom/
│   └── catalog.csv         one row per file: source, time, GPS, label
└── models/                 trained models
    ├── cnn/                .h5
    └── yolo/               .pt
```
- **Labels** (YOLO `.txt` boxes) are small, so they live in **git**, in `dataset/labels/`.
- **Empty folders** hold a 0-byte `.keep` file, because R2 can't have empty folders, so we can delete these once we have some data there in clean and models folders.

### Connect (once)
Install rclone first:

Install rclone first (Ubuntu):
```bash
sudo -v ; curl https://rclone.org/install.sh | sudo bash
rclone version          # needs 1.59 or newer for Cloudflare
```

Then:
```bash
set -a; source .env; set +a
rclone config create r2 s3 provider=Cloudflare \
  access_key_id="$R2_ACCESS_KEY_ID" secret_access_key="$R2_SECRET_ACCESS_KEY" \
  endpoint="$R2_ENDPOINT" acl=private no_check_bucket=true
rclone lsf r2:craterellus       # should list raw/ clean/ models/
```

### Download and upload
The local `dataset/` and `models/` folders mirror the bucket.
- **Run from the repo root** (`Craterellus/`), so files land in the right place.
- **`rclone copy FROM TO`:** paths starting with `r2:` are the bucket, the rest are on your local.
  - `r2:... dataset/...` downloads.
  - `dataset/... r2:...` uploads.

```bash
rclone copy r2:craterellus/clean  dataset/clean --exclude ".keep" -P   # photos + videos
rclone copy r2:craterellus/models models        --exclude ".keep" -P   # trained models
rclone copy r2:craterellus/raw    dataset/raw   -P                     # originals (pipeline only so Benjamin needs this only)

rclone copy dataset/clean r2:craterellus/clean --exclude ".gitkeep" -P   # upload
```
- **Always `copy`, never `sync` or `delete`:** `sync` deletes files that aren't on your side.
- **Don't overwrite:** new files get new names.

## Repo

```
craterellus/
├── src/
│   ├── pipeline/      Benjamin: photo, video, live pipelines + catalog
│   ├── cnn/           Eddie
│   └── yolo/          Sasha
├── app/               demo: Gradio app, Tello
├── dataset/
│   ├── raw/           ← from R2 (not in git)
│   ├── clean/         ← from R2 (not in git)
│   └── labels/        Zoi: YOLO .txt boxes (in git)
├── models/            ← from R2: cnn/, yolo/ (not in git)
├── configs/           YOLO data.yaml
├── results/           graphs, tables
├── report/            report + presentation
└── docs/              ARCHITECTURE.md, drone.md
```

## Branches

- `main`: finished work, changed only through a PR.
- `development`: features are merged here.
- `feature/*`: one branch per part.
