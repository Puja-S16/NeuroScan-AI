import tensorflow as tf
import pandas as pd

from src.unet import build_unet
from src.preprocess import get_balanced_files, load_h5


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])

    dice = (2.0 * intersection + smooth) / (denominator + smooth)

    return 1.0 - tf.reduce_mean(dice)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)

    return bce + dice_loss(y_true, y_pred)


def compile_model():
    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy")
        ]
    )

    return model


def create_balanced_tf_dataset():
    volumes = pd.read_csv("train_volumes.csv")["volume"].tolist()

    files = get_balanced_files(
        volumes,
        tumor_count=2500,
        non_tumor_count=2500
    )

    def generator():
        for path in files:
            image, mask = load_h5(path)
            yield image, mask

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec((240, 240, 4), tf.float32),
            tf.TensorSpec((240, 240, 3), tf.float32),
        )
    )

    return dataset.shuffle(1000, seed=42).batch(4).prefetch(tf.data.AUTOTUNE)


def train_model():

    train_ds = create_balanced_tf_dataset()

    val_ds = tf.data.Dataset.from_generator(
        lambda: (
            load_h5(path)
            for path in __import__("src.preprocess", fromlist=["get_slice_files"])
            .get_slice_files(
                pd.read_csv("val_volumes.csv")["volume"].tolist()
            )
        ),
        output_signature=(
            tf.TensorSpec((240, 240, 4), tf.float32),
            tf.TensorSpec((240, 240, 3), tf.float32),
        )
    ).batch(4).prefetch(tf.data.AUTOTUNE)

    model = compile_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/best_unet_small.keras",
            monitor="val_loss",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks
    )

    return model, history