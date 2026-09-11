import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint

from training.config import (
    BATCH_SIZE, CNN_MODEL_PATH, EPOCHS,
    RESNET_MODEL_PATH, SAVED_MODEL_DIR,
)
from training.data import create_data_generators
from training.models import build_cnn_model, build_resnet_model

print("GPUs available:", len(tf.config.list_physical_devices("GPU")))
SAVED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train(model: tf.keras.Model, train_gen, val_gen, save_path) -> tf.keras.callbacks.History:
    checkpoint = ModelCheckpoint(str(save_path), monitor="val_accuracy", save_best_only=True, verbose=1)
    return model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[checkpoint],
    )


def plot_history(history: tf.keras.callbacks.History) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history["accuracy"],     label="train")
    ax1.plot(history.history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.legend()
    ax2.plot(history.history["loss"],     label="train")
    ax2.plot(history.history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.legend()
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    train_gen, val_gen, _ = create_data_generators()

    cnn = build_cnn_model()
    cnn_history = train(cnn, train_gen, val_gen, CNN_MODEL_PATH)
    plot_history(cnn_history)
    print(f"CNN saved → {CNN_MODEL_PATH}")

    resnet = build_resnet_model()
    resnet_history = train(resnet, train_gen, val_gen, RESNET_MODEL_PATH)
    print(f"ResNet saved → {RESNET_MODEL_PATH}")
