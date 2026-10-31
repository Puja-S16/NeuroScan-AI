import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocess import get_slice_files, load_h5


def visualize_prediction():

    model = tf.keras.models.load_model(
        "models/best_unet_small.keras",
        compile=False
    )

    # Test set se ek slice choose karo
    test_volumes = [int(v) for v in open("test_volumes.csv").read().splitlines()[1:]]

    files = get_slice_files(test_volumes)

    # Aisa slice choose karo jisme actual tumor ho
    selected_file = None

    for path in files:
        image, mask = load_h5(path)
        if np.any(mask > 0):
            selected_file = path
            break

    if selected_file is None:
        selected_file = files[len(files) // 2]
        image, mask = load_h5(selected_file)

    prediction = model.predict(
        np.expand_dims(image, axis=0),
        verbose=0
    )[0]

    prediction = (prediction > 0.5).astype(np.uint8)

    # MRI ke first three channels ka mean display
    mri = np.mean(image[:, :, :3], axis=-1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # MRI background
    axes[0].imshow(mri, cmap="gray")
    axes[0].set_title("MRI Input")

    # Ground truth overlay
    axes[1].imshow(mri, cmap="gray")
    axes[1].imshow(
        np.any(mask > 0, axis=-1),
        cmap="Reds",
        alpha=0.65
    )
    axes[1].set_title("Ground Truth Overlay")

    # Prediction overlay
    axes[2].imshow(mri, cmap="gray")
    axes[2].imshow(
        np.any(prediction > 0, axis=-1),
        cmap="Reds",
        alpha=0.65
    )
    axes[2].set_title("U-Net Prediction Overlay")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    output_path = "results/segmentation_overlay.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Visualization saved: {output_path}")
    print(f"Source slice: {selected_file}")

    print(f"Visualization saved: {output_path}")
    print(f"Source slice: {selected_file}")


if __name__ == "__main__":
    visualize_prediction()