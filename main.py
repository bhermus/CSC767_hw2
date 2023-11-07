import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical, img_to_array

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ANIMAL_CSV = "animal_data_img.csv"
ANIMAL_IMAGES = "animal_images/"
IMAGE_HEIGHT = 107
IMAGE_WIDTH = 142
IMAGE_CHANNELS = 3


if __name__ == '__main__':
    # read in data from csv
    df = pd.read_csv(ANIMAL_CSV, encoding="utf-8", index_col=0)
    classes = sorted(df["animal_type"].unique())

    # split data into training, validation, and test sets
    training = int(len(df) * 0.7)
    validation = int(len(df) * 0.2)

    training_set = df[:training]
    validation_set = df[training:training + validation]
    test_set = df[training + validation:]

    # display class distribution
    title = "Training Set"
    print(title)
    print(training_set.value_counts("animal_type").to_string(header=False))
    plt.bar(classes, training_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    title = "Validation Set"
    print("\n" + title)
    print(validation_set.value_counts("animal_type").to_string(header=False))
    plt.bar(classes, validation_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    title = "Test Set"
    print("\n" + title)
    print(test_set.value_counts("animal_type").to_string(header=False))
    plt.bar(classes, test_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(107, 142, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D((2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Flatten())

    model.add(Dense(128, activation="relu"))

    model.add(Dropout(0.5))

    model.add(Dense(len(classes), activation="softmax"))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    # model.summary()

    images = []
    for path in training_set["image_file"].values:
        image = Image.open(ANIMAL_IMAGES + path)
        if image.size != (142, 107):
            image = image.resize((142, 107))
        images.append(image)

    X_train = np.array([img_to_array(img) for img in images])

    images = []
    for path in validation_set["image_file"].values:
        image = Image.open(ANIMAL_IMAGES + path)
        if image.size != (142, 107):
            image = image.resize((142, 107))
        images.append(image)

    X_val = np.array([img_to_array(img) for img in images])

    class_label_to_int = {class_label: i for i, class_label in enumerate(classes)}

    y_val = validation_set["animal_type"].values
    y_val = [class_label_to_int[label] for label in y_val]
    y_val_encoded = to_categorical(y_val, num_classes=len(classes))

    y_train = training_set["animal_type"].values
    y_train = [class_label_to_int[label] for label in y_train]
    y_train_encoded = to_categorical(y_train, num_classes=len(classes))

    history = model.fit(X_train, y_train_encoded, batch_size=32, epochs=30, validation_data=(X_val, y_val_encoded))
