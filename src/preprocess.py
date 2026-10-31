import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf


DATA_DIR = "/Users/pujasharma/Desktop/archive/BraTS2020_training_data/content/data"


def get_slice_files(volumes):
    """Return H5 slice paths belonging to selected volumes."""
    volume_set = set(volumes)
    files = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".h5"):
            continue

        # Example: volume_137_slice_59.h5
        parts = filename.split("_")
        volume_id = int(parts[1])

        if volume_id in volume_set:
            files.append(os.path.join(DATA_DIR, filename))

    return sorted(files)


def load_h5(path):
    """Load one H5 file."""
    

    with h5py.File(path, "r") as f:
        image = f["image"][:].astype(np.float32)
        mask = f["mask"][:].astype(np.float32)

    return image, mask


def create_dataset(volume_csv, batch_size=4, shuffle=False, max_slices=None):
    """Create a TensorFlow dataset for selected volumes."""

    volumes = pd.read_csv(volume_csv)["volume"].tolist()
    files = get_slice_files(volumes)

    if max_slices is not None:
        files = files[:max_slices]

    print(f"Volumes: {len(volumes)}")
    print(f"Slices: {len(files)}")

    def generator():
        for path in files:
            image, mask = load_h5(path)
            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(240, 240, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(240, 240, 3), dtype=tf.float32),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000, seed=42)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def has_tumor(path):
    """Check whether a slice contains any mask pixels."""

    with h5py.File(path, "r") as f:
        mask = f["mask"][:]

    return np.any(mask > 0)

def get_balanced_files(volumes, tumor_count=2500, non_tumor_count=2500):
    """Return a balanced set of tumor and non-tumor slices."""

    files = get_slice_files(volumes)

    tumor_files = []
    non_tumor_files = []

    for path in files:
        if has_tumor(path):
            tumor_files.append(path)
        else:
            non_tumor_files.append(path)

    np.random.seed(42)

    np.random.shuffle(tumor_files)
    np.random.shuffle(non_tumor_files)

    tumor_files = tumor_files[:tumor_count]
    non_tumor_files = non_tumor_files[:non_tumor_count]

    selected_files = tumor_files + non_tumor_files
    np.random.shuffle(selected_files)

    print(f"Tumor slices: {len(tumor_files)}")
    print(f"Non-tumor slices: {len(non_tumor_files)}")
    print(f"Total selected: {len(selected_files)}")

    return selected_files