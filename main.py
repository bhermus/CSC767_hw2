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


def read_csv() -> pd.DataFrame:
    df = pd.read_csv(ANIMAL_CSV, encoding="utf-8", index_col=0)
    return df


def class_counts(df: pd.DataFrame):
    training_files = os.listdir(TRAINING_IMAGES)
    validation_files = os.listdir(VALIDATION_IMAGES)
    testing_files = os.listdir(TESTING_IMAGES)

    print("TRAINING SET")
    classes = df[df["image_file"].isin(training_files)]["animal_type"].value_counts()
    print(classes.to_string(header=False))
    animal_types = classes.index.tolist()
    counts = classes.tolist()
    plt.bar(animal_types, counts)
    plt.xlabel('Animal Type')
    plt.ylabel('Count')
    plt.title('Animal Type Counts - Training')
    plt.show()

    print("\nVALIDATION SET")
    classes = df[df["image_file"].isin(validation_files)]["animal_type"].value_counts()
    print(classes.to_string(header=False))
    counts = classes.tolist()
    plt.bar(animal_types, counts)
    plt.title('Animal Type Counts - Validation')
    plt.show()

    print("\nTESTING SET")
    classes = df[df["image_file"].isin(testing_files)]["animal_type"].value_counts()
    print(classes.to_string(header=False))
    counts = classes.tolist()
    plt.bar(animal_types, counts)
    plt.title('Animal Type Counts - Testing')
    plt.show()


if __name__ == '__main__':
    df = read_csv()
    class_counts(df)
