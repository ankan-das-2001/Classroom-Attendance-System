import os

import cv2
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.io.gfile import listdir
from tensorflow.keras.preprocessing import image_dataset_from_directory

from classes.model import Model

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = "1"

# define constants
BATCH_SIZE = 16
EPOCHS = 1
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

# get class list from dataset
class_names = listdir("Datasets/train/")


def get_train_ds():
    # returns train tf.data.Dataset
    train_ds = (
        image_dataset_from_directory(
            "Datasets/train/", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
        )
        .prefetch(AUTOTUNE)
        .cache()
    )
    return train_ds


def get_val_ds():
    # returns validation tf.data.Dataset
    val_ds = image_dataset_from_directory(
        "Datasets/val/", shuffle=True, batch_size=BATCH_SIZE, image_size=IMG_SIZE
    ).prefetch(AUTOTUNE)
    return val_ds


train_ds = get_train_ds()
val_ds = get_val_ds()

model = Model(len(class_names))
model.create_model()
history = model.fit(train_ds, EPOCHS, val_ds)
model.save_model()
