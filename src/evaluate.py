import tensorflow as tf
import numpy as np
import pandas as pd

from src.preprocess import get_slice_files, load_h5


def dice_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)

    return (2 * intersection + smooth) / (
        np.sum(y_true) + np.sum(y_pred) + smooth
    )


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = y_true.astype(np.float32)
    y_pred = y_pred.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def evaluate_model():

    model = tf.keras.models.load_model(
        "models/best_unet_small.keras",
        compile=False
    )

    volumes = pd.read_csv("test_volumes.csv")["volume"].tolist()
    files = get_slice_files(volumes)

    dice_scores = []
    iou_scores = []

    for i, path in enumerate(files):

        image, mask = load_h5(path)

        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image, verbose=0)[0]
        prediction = (prediction > 0.5).astype(np.float32)

        dice_scores.append(dice_score(mask, prediction))
        iou_scores.append(iou_score(mask, prediction))

        if (i + 1) % 100 == 0:
            print(f"Evaluated: {i + 1}/{len(files)}")

    print("\nFinal Test Results")
    print("------------------")
    print(f"Mean Dice Score: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU Score:  {np.mean(iou_scores):.4f}")


if __name__ == "__main__":
    evaluate_model()