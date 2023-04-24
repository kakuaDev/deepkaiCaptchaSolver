import os
import random
import string
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Bidirectional, LSTM
from keras.layers import RepeatVector
import cv2
# from keras.regularizers import l2
# from keras.layers import BatchNormalization, Activation, LeakyReLU


class CaptchaSolver:
    CHARACTERS = " " + string.digits + string.ascii_letters
    # Total number of possible characters
    MAX_CHAR_NUM = 10
    
    def __init__(self, train_img_path, img_width=200, img_height=80, n_samples=100):
        self.train_img_path = train_img_path
        # image size
        self.img_width = img_width
        self.img_height = img_height
        self.n_samples = n_samples
        # Encoder
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(list(self.CHARACTERS))
        self.num_char_class = len(self.label_encoder.classes_)
        # Model
        self.model = self.build_model(img_width, img_height)

    def text_to_vector(self, img_path):
        text = self.get_img_name(img_path)
        # Example: 'abcd' --> ['a','b','c','d',' ',' ']
        text_vector = list(text) + (self.MAX_CHAR_NUM - len(text)) * [' ']
        # Transform the list of char into list of one-hot vectors
        return self.label_encoder.transform(text_vector)

    @staticmethod
    def get_img_name(img_path: str) -> str:
        name_plus_extension: str = img_path.split(os.path.sep)[-1]
        return name_plus_extension.split(".")[0]

    def prepare_img(self, img_file_path):
        raw_img = cv2.imread(img_file_path)
        h, w, _ = raw_img.shape
        if h != self.img_height or w != self.img_width:
            return cv2.resize(raw_img, (self.img_width, self.img_height))
        return raw_img

    def load_data(self) -> tuple:
        # sort image files
        images_files: list = random.sample(sorted(self.train_img_path), self.n_samples)
        # get image labels as names
        labels = [self.text_to_vector(img_path) for img_path in images_files]
        labels = np.asarray([
            keras.utils.to_categorical(label, self.num_char_class) for label in labels])
        images = np.array([self.prepare_img(img) for img in images_files])

        # Splitting data into training and validation sets
        x_train, x_test, y_train, y_test = train_test_split(
            images, labels, test_size=0.15)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=0.2)
        return x_train, y_train, x_valid, y_valid

    def build_model(self, img_width, img_height):
        input_img = keras.layers.Input(
            shape=(img_height, img_width, 3), name="image", dtype="float32"
        )
        x = Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3),
                   activation="relu", padding="same")(input_img)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.15)(x)

        for layer_size in [64]:
            x = Conv2D(layer_size, (3, 3), padding="same", activation="relu")(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.15)(x)

        x = Flatten()(x)
        x = Dense(64, activation="relu")(x)
        # x = Dense(512, activation="relu")(x)
        x = Dropout(0.15)(x)
        x = RepeatVector(self.MAX_CHAR_NUM)(x)
        # RNNs
        x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.15))(x)
        x = Bidirectional(LSTM(self.MAX_CHAR_NUM, return_sequences=True, dropout=0.15))(x)
        # Output layer
        x = TimeDistributed(Dense(self.num_char_class, activation="softmax", name="dense2"))(x)

        # Define the model
        model = keras.models.Model(inputs=[input_img], outputs=x)

        sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.95, nesterov=True)
        model.compile(loss=keras.losses.KLDivergence(), optimizer=sgd, metrics=['accuracy'])
        return model

    def train_model(self, model_filepath, epochs=100, load_weights=False):
        # Load dataset
        x_train, y_train, x_valid, y_valid = self.load_data()

        if load_weights:
            self.model.load_weights(model_filepath)

        save_callback = keras.callbacks.ModelCheckpoint(
            model_filepath, monitor="accuracy", save_best_only=True, save_freq="epoch"
        )
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_valid, y_valid),
            epochs=epochs,
            callbacks=[save_callback, tensorboard_callback]
        )
        return history, self.model

    def predict(self, target_img_path):
        image = self.prepare_img(target_img_path)
        prediction = self.model.predict(image)
        _, m, _ = prediction.shape
        _prediction = [np.argmax(prediction[0, i, :]) for i in range(m)]
        prediction_lists = self.label_encoder.inverse_transform(_prediction)
        return "".join([s for s in prediction_lists if s != ' '])
