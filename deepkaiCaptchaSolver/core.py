import os
import string
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, TimeDistributed
from keras.regularizers import l2
from keras.layers import BatchNormalization, RepeatVector
import cv2


CHARACTERS = string.ascii_letters + string.digits + ' '
LABEL_ENCODER = LabelEncoder()
LABEL_ENCODER.fit(list(CHARACTERS))
# Total number of possible characters
NUM_CHAR_CLASS = len(LABEL_ENCODER.classes_)
MAX_CHAR_NUM = 10
MIN_CHAR_NUM = 2


def text_to_vector(text):
    # Example: 'abcd' --> ['a','b','c','d',' ',' ']
    text_vector = list(text) + (MAX_CHAR_NUM - len(text)) * [' ']
    # Transform the list of char into list of one-hot vectors
    return LABEL_ENCODER.transform(text_vector)


def build_model(img_width, img_height):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(img_height, img_width, 3),
                     kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), kernel_initializer='he_normal', kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=l2(0.001)))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(RepeatVector(MAX_CHAR_NUM))
    # RNNs
    model.add(layers.Bidirectional(layers.LSTM(MAX_CHAR_NUM, return_sequences=True, dropout=0.25)))
    model.add(TimeDistributed(Dense(NUM_CHAR_CLASS)))
    model.add(Activation('softmax'))

    # Optimizer
    sgd = keras.optimizers.SGD(lr=0.002, momentum=0.9, nesterov=True)
    # Compile the model and return
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


class CreateModel:
    
    def __init__(self, train_img_path, img_width=200, img_height=80):
        # image size
        self.img_width = img_width
        self.img_height = img_height
        # sort image files
        images_files = sorted(train_img_path)
        # get image labels as names
        labels = [
            text_to_vector(img.split(os.path.sep)[-1].split(".png")[0])
            for img in images_files]
        self.labels = np.asarray([
            keras.utils.to_categorical(label, NUM_CHAR_CLASS) for label in labels])
        self.images = np.array([cv2.imread(img) for img in images_files])
        
    def train_model(self, epochs=100, early_stopping=False):
        # Splitting data into training and validation sets
        x_train, x_test, y_train, y_test = train_test_split(
            self.images, self.labels, test_size=0.15)
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2)
        
        # Get the model
        model = build_model(self.img_width, self.img_height)
        
        if early_stopping:
            early_stopping_patience = 10
            # Add early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience,
                restore_best_weights=True
            )
            # Train the model
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                shuffle=True,
                callbacks=[early_stopping],
            )
            model.evaluate(x_test, y_test)
        else:
            # Train the model
            model.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs,
                shuffle=True
            )
            model.evaluate(x_test, y_test)
        return model


class ApplyModel:
    
    def __init__(self, 
                 weights_path,
                 img_width=200, 
                 img_height=80):
        self.img_width = img_width
        self.img_height = img_height
        self.model = build_model(self.img_width, self.img_height)
        self.model.load_weights(weights_path)
    
    def predict(self, target_img_path):
        image = cv2.imread(target_img_path)
        target_img: np.array = cv2.resize(
            image, (self.img_width, self.img_height), interpolation=cv2.INTER_AREA)
        target_img = target_img.reshape(1, self.img_height, self.img_width, 3)
        prediction = self.model.predict(target_img)
        _, m, _ = prediction.shape
        _prediction = [np.argmax(prediction[0, i, :]) for i in range(m)]
        prediction_lists = LABEL_ENCODER.inverse_transform(_prediction)
        return "".join([s for s in prediction_lists if s != ' '])
