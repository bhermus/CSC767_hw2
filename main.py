import os
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ANIMAL_CSV = "animal_data_img.csv"
ANIMAL_IMAGES = "animal_images/"
TRAINING_IMAGES = ANIMAL_IMAGES + "training/"
VALIDATION_IMAGES = ANIMAL_IMAGES + "validation/"
TESTING_IMAGES = ANIMAL_IMAGES + "testing/"


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
    plt.show()

    title = "Validation Set"
    print("\n" + title)
    print(validation_set.value_counts("animal_type").to_string(header=False))
    plt.bar(classes, validation_set.value_counts("animal_type"))
    plt.title(title)
    plt.show()

    title = "Test Set"
    print("\n" + title)
    print(test_set.value_counts("animal_type").to_string(header=False))
    plt.bar(classes, test_set.value_counts("animal_type"))
    plt.title(title)
    plt.show()
