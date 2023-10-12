import glob
import os
import string
import unicodedata

from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words("english"))


def preprocess_string(text: str) -> str:
    # Following the order of the slides in the lecture

    text = text.replace("\n", " ").strip()  # Remove newlines and trailing whitespace
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove puctuation with lookup table
    text = text.lower()  # Lowercase
    text = " ".join([word for word in text.split() if word not in STOP_WORDS])  # Remove stopwords
    text = text.translate(str.maketrans("", "", string.digits))  # Remove all numbers with lookup table

    # Remove excess whitespace in between words
    # E.g. the sentence "for 10 days" becomes "for days" instead of "for  days" with two spaces
    text = " ".join(text.split())
    text = unicodedata.normalize("NFKD", text)  # Strip accents from characters

    # TODO: Lemmatization

    return text


def iterate_over_filepaths(fp_data: str, glob_pattern: str = "*.txt") -> list:
    fps = glob.glob(os.path.join(fp_data, glob_pattern))  # Get all txt files using glob
    docs = []

    for fp in fps:
        with open(fp, "r", encoding="utf-8") as file:
            text = file.read()

        text = preprocess_string(text)
        docs.append(text)

    return docs


def iterate_over_folds(fp_data: str, folds_train: list = ["fold1", "fold2", "fold3", "fold4"], folds_test: list = ["fold5"]) -> tuple:
    docs_train = []
    docs_test = []

    for fold in folds_train:
        fp_fold = os.path.join(fp_data, fold)
        docs_train.extend(iterate_over_filepaths(fp_fold))

    for fold in folds_test:
        fp_fold = os.path.join(fp_data, fold)
        docs_test.extend(iterate_over_filepaths(fp_fold))

    assert len(folds_train) + len(folds_test) == 5, f"Number of folds must be 5, but is {len(folds_train)} + {len(folds_test)}"
    assert len(docs_train) + len(docs_test) == 400, f"Number of documents must be 400, but is {len(docs_train)} + {len(docs_test)}"

    return docs_train, docs_test


def load_data(fp_data, folds_train: list = ["fold1", "fold2", "fold3", "fold4"], folds_test: list = ["fold5"]) -> tuple:
    # Load deceptive
    fp_deceptive = os.path.join(fp_data, "deceptive_from_MTurk")
    docs_deceptive_train, docs_deceptive_test = iterate_over_folds(fp_deceptive, folds_train, folds_test)

    # Load truthfull
    fp_truthfull = os.path.join(fp_data, "truthful_from_Web")
    docs_truthfull_train, docs_truthfull_test = iterate_over_folds(fp_truthfull, folds_train, folds_test)

    return docs_deceptive_train, docs_deceptive_test, docs_truthfull_train, docs_truthfull_test


if __name__ == "__main__":
    fp_data = "./op_spam_v1.4/negative_polarity/"

    deceptive_train, deceptive_test, truthfull_train, truthfull_test = load_data(fp_data)

    print(deceptive_train[0])
    print(f"Deceptive train: {len(deceptive_train)}")
    print(f"Deceptive test: {len(deceptive_test)}")
    print(f"Truthfull train: {len(truthfull_train)}")
    print(f"Truthfull test: {len(truthfull_test)}")
