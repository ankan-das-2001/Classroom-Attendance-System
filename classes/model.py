import tensorflow as tf
import tensorflow.keras as keras
from cv2 import resize
from numpy import argmax, expand_dims


# define model
class Model:
    __checkpoint_path = "training/cp.ckpt"
    __device = tf.test.gpu_device_name()
    if __device == "":
        __device = "/cpu:0"
    with tf.device(__device):
        # define private layers, checkpoint dir, and callbacks
        __preprocess_input = keras.applications.mobilenet_v2.preprocess_input
        __callbacks = [
            keras.callbacks.EarlyStopping("val_acc", 0.001, 3),
            keras.callbacks.ModelCheckpoint(
                __checkpoint_path, save_best_only=True, save_weights_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(monitor="acc", patience=3),
        ]

    def __init__(self, len_class, IMG_SHAPE=(224, 224, 3)):
        self.__IMG_SHAPE = IMG_SHAPE
        self.__len_class = len_class

    def create_model(self, flip="horizontal", rotation=(-0.3, 0.3)):
        with tf.device(self.__device):
            # define augmentations
            self.__aug = keras.Sequential(
                [
                    keras.layers.experimental.preprocessing.RandomFlip(flip),
                    keras.layers.experimental.preprocessing.RandomRotation(rotation),
                ]
            )

            # load base model into memory
            self.__base = keras.applications.MobileNetV2(
                self.__IMG_SHAPE, include_top=False
            )
            self.__base.trainable = False

            # define face recognition model
            self.__inputs = keras.layers.Input(self.__IMG_SHAPE)
            self.__x = self.__aug(self.__inputs)
            self.__x = keras.applications.mobilenet_v2.preprocess_input(self.__x)
            self.__x = self.__base(self.__x)
            self.__x = keras.layers.GlobalAveragePooling2D()(self.__x)
            self.__x = keras.layers.Dropout(0.3)(self.__x)
            self.__x = keras.layers.Dense(512, activation="relu")(self.__x)
            self.__x = keras.layers.Dropout(0.3)(self.__x)
            self.__x = keras.layers.Dense(256, activation="relu")(self.__x)
            self.__outputs = keras.layers.Dense(self.__len_class, activation="softmax")(
                self.__x
            )

            self.__model = keras.Model(self.__inputs, self.__outputs)
        self.__model.compile(
            optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"]
        )

    def fit(self, train_ds, epochs=10, val_ds=None):
        # fit model on tf.data.Dataset
        history = self.__model.fit(
            train_ds, epochs=epochs, validation_data=val_ds, callbacks=self.__callbacks
        )
        self.__model.load_weights(self.__checkpoint_path)
        return history

    def save_model(self, dir="savedmodel/"):
        # save model in SavedModel format
        self.__model.save(dir)

    def load_model(self, dir="savedmodel/"):
        # load model from SavedModel
        self.__model = tf.keras.models.load_model(dir)

    def predict(self, img):
        # predict on tensor or tf.data.Dataset
        return argmax(self.__model.predict(img), axis=1)

    def predict_on_cv(self, img):
        # predict on OpenCV image
        temp = resize(img, (224, 224))
        temp = tf.convert_to_tensor(temp, dtype=tf.float32)
        temp = tf.expand_dims(temp, axis=0)
        return self.predict(temp)
