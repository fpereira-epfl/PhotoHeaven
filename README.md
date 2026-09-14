# PhotoHeaven

PhotoHeaven is a local-first Python CLI for organising, analysing, and
searching personal photo and video libraries. Everything runs on your own
machine; metadata is stored in a local SQLite database inside your library.

What it can do:

- Import and catalogue photos and videos.
- Detect, cluster, and name faces.
- Detect country (from GPS) and scene (using Places365).
- Search by person, date, country, and scene.
- Find duplicate photos and videos.
- Rename and reorganise files by capture date.

## Requirements

- Python 3.11 or newer.
- The `MediaInfo` command-line tool for video/container metadata. Install it
  with your system package manager, e.g.:
  - macOS: `brew install mediainfo`
  - Ubuntu/Debian: `sudo apt install mediainfo`

## Installation

```bash
git clone <repository-url>
cd PhotoHeaven
python -m venv .venv.photoheaven
source .venv.photoheaven/bin/activate
pip install -e .
```

This installs the `ph` and `photoheaven` commands in the active environment.

## Quick start

Tell PhotoHeaven where your library lives. You can either export an
environment variable or pass `--library` to every command:

```bash
export PHOTOHEAVEN_LIBRARY=/path/to/PhotoHeaven.photoslibrary
```

Create a new self-contained library:

```bash
ph init
```

Import some photos:

```bash
# Copy photos into the library
ph import /path/to/photos --recursive

# Or move them instead of copying
ph import /path/to/photos --recursive --move
```

Build the database index:

```bash
ph sync
```

## Library layout

A PhotoHeaven library is a single folder that contains everything:

```
<library>/
  db/
    photoheaven.db          # SQLite metadata database
  files/
    YYYY/MM/                # imported media files
  duplicates/
    YYYY/MM/                # moved duplicates (created by dedupe)
```

Because the database lives next to the files, the whole library can be moved
or backed up as one unit.

## Face workflow

Detect faces and compute embeddings:

```bash
ph faces detect
```

Cluster the faces into identity groups:

```bash
ph faces cluster
```

List the largest clusters:

```bash
ph faces list
```

Assign a human name to a cluster:

```bash
ph faces name <cluster_label> "Francisco"
```

Automatically label remaining faces by similarity to named identities:

```bash
ph faces assign
```

Show sample photos for a cluster:

```bash
ph faces samples <cluster_label>
```

## Place detection

Detect country and scene for unprocessed images:

```bash
ph places detect
```

Useful options:

```bash
# Only analyse 50 random photos
ph places detect --limit 50 --random

# Require a higher scene confidence before storing a label
ph places detect --scene-threshold 0.5

# Re-analyse everything
ph places detect --force

# Clear all place results
ph places reset -y
```

Notes:

- **Country** is derived from EXIF GPS only. If a photo has no GPS tags, the
  country is left blank.
- **Scene** uses a Places365 CNN. The default confidence threshold is `0.1`;
  increase it to keep only stronger predictions.
- The first run downloads the Places365 model (~95 MB) and category list to
  `~/.cache/photoheaven/places365/`.

## Search

Search by one or more criteria:

```bash
# Photos containing both Francisco and Katerina
ph search -n Francisco,Katerina

# Photos from 2023
ph search -Y 2023

# Photos from May 2023
ph search -Y 2023 -M 5

# Photos taken between dates
ph search --from 2023-01 --to 2023-06

# Photos classified as a beach scene
ph search -s beach

# Photos in Portugal
ph search -c Portugal

# Combine filters
ph search -n Francisco,Katerina -s waterfall -Y 2023
```

Other useful options:

```bash
# Include videos
ph search --include-videos

# Show more results
ph search -n Francisco --limit 500

# List every distinct country and scene available for search
ph search --list-places
```

Notes:

- Comma-separated names in `-n/--names` are treated as **AND**: every listed
  person must be present in the photo.
- `--country` and `--scene` are case-insensitive.
- Search results show the detected persons and place under each file path.

## Duplicate detection

Scan for duplicates using checksum and perceptual hash:

```bash
ph dedupe
```

Move non-primary duplicates to `<library>/duplicates/YYYY/MM`:

```bash
ph dedupe --move
```

List already-known duplicate groups:

```bash
ph dedupe --list
```

Options:

- `--max-distance`: perceptual-hash threshold (default 5).
- `--include-videos`: also detect duplicate videos.
- `--dry-run`: preview changes without moving files.
- `--archive <path>`: archive everything under `<library>/duplicates` to an
  external folder.

## Rename and organise

Rename files to `YYYY-MM-DD_HHhMMmSSs.<ext>` based on capture date:

```bash
# Preview
ph rename --move --dry-run

# Apply
ph rename --move
```

Append recognised face names to filenames:

```bash
ph rename --include-faces --move
```

## Maintenance commands

```bash
ph info                          # Library statistics
ph inspect                       # Inspect metadata for all files
ph inspect -i <file>             # Inspect a single file
ph sync --prune                  # Remove DB records for missing files
ph clean                         # Remove empty folders under files/
ph rebase --dry-run              # Rebase stored paths to the library root
```

## Privacy and data notes

- Face embeddings, GPS coordinates, and other metadata are stored locally in
  `<library>/db/photoheaven.db`.
- The database uses SQLite WAL mode for resilience, which is helpful if the
  library folder lives on a cloud-synced volume.
- PhotoHeaven does not send your files or metadata to any external service.
  Pre-trained models are downloaded from Hugging Face and the MIT CSAIL
  Places365 project on first use.

## Shell completion

Install tab completion for your current shell:

```bash
ph --install-completion
```
