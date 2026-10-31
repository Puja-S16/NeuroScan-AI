
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf

EXAMPLE_DIR = "data/demo_examples"
MODEL_PATH = "models/best_unet_small.keras"

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_examples():
    manifest_path = os.path.join(EXAMPLE_DIR, "manifest.csv")
    manifest = pd.read_csv(manifest_path)

    examples = []

    for _, row in manifest.iterrows():
        path = os.path.join(EXAMPLE_DIR, row["file"])

        with np.load(path, allow_pickle=False) as data:
            examples.append({
                "label": (
                    f"Example {int(row['example']):02d} — "
                    f"{row['source']}"
                ),
                "image": data["image"].copy(),
                "mask": data["mask"].copy(),
                "source": str(row["source"]),
                "volume": int(row["volume"]),
                "slice": int(row["slice"])
            })

    return examples


def calculate_metrics(y_true, y_pred):
    true = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)

    intersection = np.logical_and(true, pred).sum()
    union = np.logical_or(true, pred).sum()

    dice = (
        2.0 * intersection /
        (true.sum() + pred.sum() + 1e-8)
    )
    iou = intersection / (union + 1e-8)

    return float(dice), float(iou)


st.title("🧠 NeuroScan AI")
st.subheader("Brain MRI Tumor Segmentation")

st.write(
    "Explore 25 built-in MRI examples and compare the "
    "trained U-Net prediction with the reference mask."
)

st.warning(
    "Educational research prototype only. "
    "Predictions are not a medical diagnosis and must not "
    "be used for clinical decisions."
)

try:
    model = load_model()
    examples = load_examples()

    selected_label = st.selectbox(
        "Choose an MRI example",
        [item["label"] for item in examples]
    )

    sample = next(
        item for item in examples
        if item["label"] == selected_label
    )

    st.caption(
        f"Source: {sample['source']} | "
        f"Volume: {sample['volume']} | "
        f"Slice: {sample['slice']}"
    )

    if st.button("🔍 Run Segmentation", type="primary"):
        with st.spinner("Running U-Net..."):
            image = sample["image"]
            mask = sample["mask"]

            prediction_prob = model.predict(
                image[np.newaxis, ...],
                verbose=0
            )[0]

            prediction = (prediction_prob >= 0.5).astype(np.uint8)

        # Binary union across the three mask channels.
        ground_truth = np.any(mask > 0, axis=-1)
        predicted = np.any(prediction > 0, axis=-1)

        dice, iou = calculate_metrics(
            ground_truth, predicted
        )

        # Display-only grayscale composite of three input channels.
        mri = np.mean(image[:, :, :3], axis=-1)

        coverage = 100.0 * predicted.mean()
        gt_coverage = 100.0 * ground_truth.mean()

        st.success("Segmentation complete")

        c1, c2, c3 = st.columns(3)
        c1.metric("Dice Score", f"{dice:.4f}")
        c2.metric("IoU Score", f"{iou:.4f}")
        c3.metric("Predicted Mask Coverage", f"{coverage:.2f}%")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(mri, cmap="gray")
        axes[0].set_title("MRI Input")

        axes[1].imshow(mri, cmap="gray")
        axes[1].imshow(
            ground_truth,
            cmap="Reds",
            alpha=0.65,
            vmin=0,
            vmax=1
        )
        axes[1].set_title("Reference Mask")

        axes[2].imshow(mri, cmap="gray")
        axes[2].imshow(
            predicted,
            cmap="Reds",
            alpha=0.65,
            vmin=0,
            vmax=1
        )
        axes[2].set_title("U-Net Prediction")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Generate an educational research summary.
        report = f"""
NEUROSCAN AI
MRI SEGMENTATION RESEARCH SUMMARY
================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Sample: {sample['source']}
Test volume: {sample['volume']}
Slice: {sample['slice']}

MODEL
U-Net
Threshold: 0.50

SEGMENTATION RESULTS
Binary union Dice score: {dice:.4f}
Binary union IoU score: {iou:.4f}
Predicted mask coverage: {coverage:.4f}%
Reference mask coverage: {gt_coverage:.4f}%

INTERPRETATION
Dice and IoU measure overlap between the model's predicted
binary mask and the dataset's reference mask for this slice.
Mask coverage is the percentage of image pixels marked by
the predicted segmentation.

LIMITATIONS
This is an educational research prototype, not a clinical
report or diagnostic tool. Results apply to this sample only.
A predicted mask does not establish the presence, type, grade,
or clinical significance of a brain tumor. Expert review and
appropriate clinical assessment are required for medical use.
The displayed grayscale MRI is a visualization composite of
the first three input channels, not a clinical image sequence.
"""

        os.makedirs("reports", exist_ok=True)

        report_filename = (
            f"neuroscan_report_example_{sample['volume']}_"
            f"{sample['slice']}.txt"
        )

        report_path = os.path.join("reports", report_filename)

        with open(report_path, "w", encoding="utf-8") as file:
            file.write(report.strip() + "\n")

        st.divider()
        st.header("📄 Segmentation Research Report")
        st.text(report)

        st.download_button(
            "⬇️ Download Report",
            data=report.strip() + "\n",
            file_name=report_filename,
            mime="text/plain"
        )

except FileNotFoundError as error:
    st.error(
        f"Required file not found: {error}. "
        "Run the example export script first."
    )

except Exception as error:
    st.error(f"Something went wrong: {error}")
