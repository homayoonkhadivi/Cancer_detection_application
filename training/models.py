import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Dense, Dropout,
    Flatten, GlobalAveragePooling2D, MaxPool2D,
)
from tensorflow.keras.models import Sequential

from training.config import IMG_HEIGHT, IMG_WIDTH


def build_cnn_model() -> tf.keras.Model:
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2)),

        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2)),

        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2)),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_resnet_model() -> tf.keras.Model:
    base = tf.keras.applications.ResNet50(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights="imagenet"
    )
    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"), BatchNormalization(), Dropout(0.6),
        Dense(128, activation="relu"), BatchNormalization(), Dropout(0.4),
        Dense(64,  activation="relu"), BatchNormalization(), Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )
    return model
