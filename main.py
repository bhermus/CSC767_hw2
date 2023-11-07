import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from keras import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from keras.src.utils import to_categorical, img_to_array

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ANIMAL_CSV = "animal_data_img.csv"
ANIMAL_IMAGES = "animal_images/"
IMAGE_HEIGHT = 107
IMAGE_WIDTH = 142
IMAGE_CHANNELS = 3

# plt.show(block=False)
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
    plt.figure()
    plt.bar(classes, training_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    title = "Validation Set"
    print("\n" + title)
    print(validation_set.value_counts("animal_type").to_string(header=False))
    plt.figure()
    plt.bar(classes, validation_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    title = "Test Set"
    print("\n" + title)
    print(test_set.value_counts("animal_type").to_string(header=False))
    plt.figure()
    plt.bar(classes, test_set.value_counts("animal_type"))
    plt.title(title)
    # plt.show()

    # organize our datasets into X and y labels so we can run them through the CNN
    images = []
    for path in training_set["image_file"].values:
        image = Image.open(ANIMAL_IMAGES + path)
        if image.size != (142, 107):  # as a precaution, we're making sure all images are the same size
            image = image.resize((142, 107))
        images.append(image)

    X_train = np.array([img_to_array(img) for img in images])  # this is our training input

    images = []
    for path in validation_set["image_file"].values:
        image = Image.open(ANIMAL_IMAGES + path)
        if image.size != (142, 107):
            image = image.resize((142, 107))
        images.append(image)

    X_val = np.array([img_to_array(img) for img in images])  # this is our validation input

    class_label_to_int = {class_label: i for i, class_label in enumerate(classes)}

    y_val = validation_set["animal_type"].values
    y_val = [class_label_to_int[label] for label in y_val]
    y_val_encoded = to_categorical(y_val, num_classes=len(classes))  # these are our validation labels

    y_train = training_set["animal_type"].values
    y_train = [class_label_to_int[label] for label in y_train]
    y_train_encoded = to_categorical(y_train, num_classes=len(classes))  # these are our training labels

    # augment our training data to help balance it
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    rabbit_class_label = class_label_to_int['Rabbit']
    rabbit_indices = [i for i, label in enumerate(y_train) if label == rabbit_class_label]
    num_augmented_samples = 2300
    augmented_samples = []
    for i in range(num_augmented_samples):
        random_index = np.random.choice(rabbit_indices)
        sample = X_train[random_index]
        augmented_sample = datagen.random_transform(X_train[random_index])
        augmented_samples.append(augmented_sample)

    augmented_samples = np.array(augmented_samples).reshape(-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

    X_train_balanced = np.array(X_train.tolist() + augmented_samples.tolist())
    y_train_balanced = np.hstack([y_train, [class_label_to_int['Rabbit']] * num_augmented_samples])

    # display balanced training set breakdown
    title = "Balanced Training Set"
    print("\n" + title)
    print(pd.Series(y_train_balanced).value_counts().sort_index().to_string(header=False))
    plt.figure()
    plt.bar(classes, pd.Series(y_train_balanced).value_counts().sort_index())
    plt.title(title)
    plt.show()

    # set up our CNN
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

    model.summary()

    # fit our CNN to our unbalanced training set, with our provided validation set
    history = model.fit(X_train, y_train_encoded, batch_size=32, epochs=30, validation_data=(X_val, y_val_encoded))

    # get training and validation loss and accuracy values from history
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    # using the same CNN on the balanced training set
    y_train_balanced_encoded = to_categorical(y_train_balanced, num_classes=len(classes))
    print(X_train_balanced.shape)
    print(y_train_balanced_encoded.shape)
    history = model.fit(X_train_balanced, y_train_balanced_encoded, batch_size=32, epochs=30, validation_data=(X_val, y_val_encoded))

    # get training and validation loss and accuracy values from history
    loss_balanced = history.history['loss']
    val_loss_balanced = history.history['val_loss']
    accuracy_balanced = history.history['accuracy']
    val_accuracy_balanced = history.history['val_accuracy']

    # create epochs range
    epochs = range(1, len(loss) + 1)

    # plot training loss
    plt.figure()
    plt.plot(epochs, loss, 'b', label='Unbalanced')
    plt.plot(epochs, loss_balanced, 'r', label='Balanced')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    # plot validation loss
    plt.figure()
    plt.plot(epochs, val_loss, 'b', label='Unbalanced')
    plt.plot(epochs, val_loss_balanced, 'r', label='Balanced')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()

    # plot training accuracy
    plt.figure()
    plt.plot(epochs, accuracy, 'b', label='Unbalanced')
    plt.plot(epochs, accuracy_balanced, 'r', label='Balanced')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # plt.show()

    # plot validation accuracy
    plt.figure()
    plt.plot(epochs, val_accuracy, 'b', label='Unbalanced')
    plt.plot(epochs, val_accuracy_balanced, 'r', label='Balanced')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    