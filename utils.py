import glob
import os
import string
import unicodedata

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

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


def vectorize_data(vectorizer: CountVectorizer, class_0_train: list, class_0_test: list,
                   class_1_train: list, class_1_test: list) -> tuple:
    # Fit and transform the training data
    # 0: deceptive, 1: truthfull
    X_train = vectorizer.fit_transform(class_0_train + class_1_train).toarray()
    y_train = [0] * len(class_0_train) + [1] * len(class_1_train)

    # Only transform the test data
    X_test = vectorizer.transform(class_0_test + class_1_test).toarray()
    y_test = [0] * len(class_0_test) + [1] * len(class_1_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    fp_data = "./op_spam_v1.4/negative_polarity/"
    max_features = None  # Maximum vocab size
    ngram_range = (1, 1)  # Range of n-grams to include in the vocabulary
    min_df = 0.05  # Minimal document frequency of a word to be included in the vocabulary

    deceptive_train, deceptive_test, truthfull_train, truthfull_test = load_data(fp_data)

    print(f"Deceptive train: {len(deceptive_train)}")
    print(f"Deceptive test: {len(deceptive_test)}")
    print(f"Truthfull train: {len(truthfull_train)}")
    print(f"Truthfull test: {len(truthfull_test)}")

    vectorizer = CountVectorizer(analyzer="word", max_features=max_features, ngram_range=ngram_range)

    X_train, X_test, y_train, y_test = vectorize_data(vectorizer,
                                                      deceptive_train, deceptive_test,
                                                      truthfull_train, truthfull_test)

    # Split train into train and validation, stratified
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=42)

    print(f"X_train: {X_train.shape}; y_train: {len(y_train)}")
    print(f"X_val: {X_val.shape}; y_val: {len(y_val)}")
    print(f"X_test: {X_test.shape}; y_test: {len(y_test)}")

    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # Convert array to list of words
    example = X_train[0].tolist()
    print(example[:20])
    # List of all words in the vocabulary that are present in the example
    words = vectorizer.inverse_transform([example])[0]
    print(f"In total {len(words)} words are present in the example. "
          f"Excerpt of 20 words: {' '.join(words[:20])}")
