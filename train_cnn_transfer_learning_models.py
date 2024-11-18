import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
import tensorflow as tf

# Check GPU availability
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    print("Using GPU for training.")
else:
    print("GPU not available. Using CPU for training.")

# Dataset directories
TRAIN_DIR = "./input/chest_xray/chest_xray/train"
VAL_DIR = "./input/chest_xray/chest_xray/val"
TEST_DIR = "./input/chest_xray/chest_xray/test"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 180, 180
BATCH_SIZE = 32
EPOCHS = 3
MODEL_SAVE_PATH_CNN = "./saved_model/chest_xray_model.keras"
MODEL_SAVE_PATH_RESNET = "./saved_model/resnet_model.keras"

# Data Generators
def create_data_generators():
    """
    Creates data generators for training, validation, and testing datasets with data augmentation.
    """
    image_generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        samplewise_center=True,
        samplewise_std_normalization=True
    )

    train_gen = image_generator.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    val_gen = image_generator.flow_from_directory(
        VAL_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_gen = image_generator.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False  # No shuffling for testing
    )
    return train_gen, val_gen, test_gen

# Build CNN Model
def build_cnn_model():
    """
    Builds and compiles a CNN model for binary classification.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Build ResNet Model
def build_resnet_model():
    """
    Builds and compiles a ResNet-based model for binary classification.
    """
    resnet_base_model = ResNet50(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
    model = tf.keras.Sequential([
        resnet_base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        BatchNormalization(),
        Dropout(0.6),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    return model

# Train Model
def train_model(model, train_gen, val_gen, save_path):
    """
    Trains the model and saves the best weights.
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    checkpoint = ModelCheckpoint(
        save_path, 
        monitor='val_accuracy', 
        save_best_only=True, 
        verbose=1
    )

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_steps=val_gen.samples // BATCH_SIZE,
        callbacks=[checkpoint]
    )
    return history

# Plot Training History
def plot_training_history(history):
    """
    Plots training and validation accuracy/loss.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.show()

if __name__ == "__main__":
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()

    # Train CNN model
    cnn_model = build_cnn_model()
    cnn_history = train_model(cnn_model, train_gen, val_gen, MODEL_SAVE_PATH_CNN)

    # Train ResNet model
    resnet_model = build_resnet_model()
    resnet_history = train_model(resnet_model, train_gen, val_gen, MODEL_SAVE_PATH_RESNET)

    # Plot training history for CNN
    plot_training_history(cnn_history)

    print(f"CNN Model saved at {MODEL_SAVE_PATH_CNN}")
    print(f"ResNet Model saved at {MODEL_SAVE_PATH_RESNET}")
