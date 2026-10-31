
import os
import random
import numpy as np
import pandas as pd

from src.preprocess import get_slice_files, load_h5

OUTPUT_DIR = "data/demo_examples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

rng = random.Random(2026)

volumes = pd.read_csv("test_volumes.csv")["volume"].tolist()
rng.shuffle(volumes)

selected = []

# Pick tumor-containing slices from different test volumes.
for volume in volumes:
    files = get_slice_files([int(volume)])
    tumor_files = []

    for path in files:
        _, mask = load_h5(path)
        if np.any(mask > 0):
            tumor_files.append(path)

    if tumor_files:
        selected.append(rng.choice(tumor_files))

    if len(selected) == 25:
        break

if len(selected) < 25:
    raise RuntimeError(
        f"Only found {len(selected)} suitable examples, not 25."
    )

manifest = []

for i, path in enumerate(selected, start=1):
    image, mask = load_h5(path)
    filename = f"example_{i:02d}.npz"

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, filename),
        image=image.astype(np.float32),
        mask=mask.astype(np.uint8),
        sample_id=np.array(os.path.basename(path))
    )

    manifest.append({
        "example": i,
        "file": filename,
        "source": os.path.basename(path),
        "volume": int(path.split("volume_")[1].split("_")[0]),
        "slice": int(path.split("_slice_")[1].split(".")[0])
    })

pd.DataFrame(manifest).to_csv(
    os.path.join(OUTPUT_DIR, "manifest.csv"),
    index=False
)

print(f"Exported {len(selected)} examples to {OUTPUT_DIR}")

for item in manifest:
    print(
        f"Example {item['example']:02d}: "
        f"{item['source']}"
    )
