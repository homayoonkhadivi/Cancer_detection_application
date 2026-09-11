from tensorflow.keras.preprocessing.image import ImageDataGenerator

from training.config import BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, TEST_DIR, TRAIN_DIR, VAL_DIR


def create_data_generators():
    gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        samplewise_center=True,
        samplewise_std_normalization=True,
    )

    kwargs = dict(target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="binary")

    train_gen = gen.flow_from_directory(str(TRAIN_DIR), **kwargs)
    val_gen   = gen.flow_from_directory(str(VAL_DIR), **kwargs)
    test_gen  = gen.flow_from_directory(str(TEST_DIR), shuffle=False, **kwargs)

    return train_gen, val_gen, test_gen
