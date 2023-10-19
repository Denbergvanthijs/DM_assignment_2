import glob
import os
import string
import unicodedata

import nltk
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


def preprocess_string(text: str, stop_words: set = set(stopwords.words("english")), unicode_pattern: str = "NFKD") -> str:
    # Following the order of the slides in the lecture

    text = text.replace("\n", " ").strip()  # Remove newlines and trailing whitespace
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove puctuation with lookup table
    text = text.lower()  # Lowercase
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    text = text.translate(str.maketrans("", "", string.digits))  # Remove all numbers with lookup table

    # Remove excess whitespace in between words
    # E.g. the sentence "for 10 days" becomes "for days" instead of "for  days" with two spaces
    text = " ".join(text.split())
    text = unicodedata.normalize(unicode_pattern, text)  # Strip accents from characters

    # Lemmatization
    lemmanizer = nltk.stem.WordNetLemmatizer()
    text = " ".join([lemmanizer.lemmatize(word) for word in text.split()])

    # TODO: Combine multiple "".join to one for speedup

    return text


def iterate_over_filepaths(fp_data: str, glob_pattern: str = "*.txt",
                           stop_words: set = set(stopwords.words("english")), unicode_pattern: str = "NFKD") -> list:
    fps = glob.glob(os.path.join(fp_data, glob_pattern))  # Get all txt files using glob
    docs = []

    for fp in fps:
        with open(fp, "r", encoding="utf-8") as file:
            text = file.read()

        text = preprocess_string(text, stop_words, unicode_pattern)
        docs.append(text)

    return docs


def iterate_over_folds(fp_data: str, folds_train: list = ["fold1", "fold2", "fold3", "fold4"],
                       folds_test: list = ["fold5"], glob_pattern: str = "*.txt",
                       stop_words: set = set(stopwords.words("english")), unicode_pattern: str = "NFKD") -> tuple:
    docs_train = []
    docs_test = []

    for fold in folds_train:
        fp_fold = os.path.join(fp_data, fold)
        docs_train.extend(iterate_over_filepaths(fp_fold, glob_pattern, stop_words, unicode_pattern))

    for fold in folds_test:
        fp_fold = os.path.join(fp_data, fold)
        docs_test.extend(iterate_over_filepaths(fp_fold, glob_pattern, stop_words, unicode_pattern))

    assert len(folds_train) + len(folds_test) == 5, f"Number of folds must be 5, but is {len(folds_train)} + {len(folds_test)}"
    assert len(docs_train) + len(docs_test) == 400, f"Number of documents must be 400, but is {len(docs_train)} + {len(docs_test)}"

    return docs_train, docs_test


def load_data(fp_data, folds_train: list = ["fold1", "fold2", "fold3", "fold4"],
              folds_test: list = ["fold5"], glob_pattern: str = "*.txt",
              stop_words: set = set(stopwords.words("english")), unicode_pattern: str = "NFKD") -> tuple:
    # Load deceptive
    fp_deceptive = os.path.join(fp_data, "deceptive_from_MTurk")
    docs_deceptive_train, docs_deceptive_test = iterate_over_folds(fp_deceptive, folds_train, folds_test,
                                                                   glob_pattern, stop_words, unicode_pattern)

    # Load truthfull
    fp_truthfull = os.path.join(fp_data, "truthful_from_Web")
    docs_truthfull_train, docs_truthfull_test = iterate_over_folds(fp_truthfull, folds_train, folds_test,
                                                                   glob_pattern, stop_words, unicode_pattern)

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


def pipeline(fp_data: str, max_features: int = None, ngram_range: tuple = (1, 1), min_df: float = 0.05,
             folds_train: list = ["fold1", "fold2", "fold3", "fold4"], folds_test: list = ["fold5"],
             glob_pattern: str = "*.txt", stop_words: set = set(stopwords.words("english")), val_size: float = None,
             unicode_pattern: str = "NFKD", random_state: int = 42) -> tuple:
    """Complete pipeline for loading and vectorizing the data

    :param fp_data: Filepath to the data folder containing the folds
    :type fp_data: str
    :param max_features: Max number of features to include in the vocabulary, defaults to None
    :type max_features: int, optional
    :param ngram_range: Range of n-grams to include in the vocabulary, defaults to (1, 1)
    :type ngram_range: tuple, optional
    :param min_df: Minimal document frequency of a word to be included in the vocabulary, defaults to 0.05
    :type min_df: float, optional
    :param folds_train: Folds to be used for training, defaults to ["fold1", "fold2", "fold3", "fold4"]
    :type folds_train: list, optional
    :param folds_test: Folds to be used for testing, defaults to ["fold5"]
    :type folds_test: list, optional
    :param glob_pattern: Glob pattern to use for loading the data, defaults to "*.txt"
    :type glob_pattern: str, optional
    :param stop_words: Set of stopwords to use for preprocessing, defaults to set(stopwords.words("english"))
    :type stop_words: set, optional
    :param unicode_pattern: Unicode pattern to use for preprocessing, defaults to "NFKD"
    :type unicode_pattern: str, optional
    :param val_size: Size of the validation set as a percentage of the training set, defaults to 0.2
    :type val_size: float, optional
    :param random_state: Random state to use for splitting the data, defaults to 42
    :type random_state: int, optional

    :return: X_train, X_val, X_test, y_train, y_val, y_test, vectorizer
    :rtype: tuple
    """
    # Load the data and preprocess it
    deceptive_train, deceptive_test, truthfull_train, truthfull_test = load_data(fp_data, folds_train, folds_test,
                                                                                 glob_pattern, stop_words, unicode_pattern)

    # Initialize the vectorizer
    vectorizer = CountVectorizer(analyzer="word", max_features=max_features, ngram_range=ngram_range, min_df=min_df)

    # Vectorize the data
    X_train, X_test, y_train, y_test = vectorize_data(vectorizer,
                                                      deceptive_train, deceptive_test,
                                                      truthfull_train, truthfull_test)

    if val_size is None:
        return X_train, X_test, y_train, y_test, vectorizer

    # Split train into train and validation, stratified
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                      test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Calculate accuracy, precision, recall and f1 score

    :param y_true: Binary ground truth labels
    :type y_true: np.ndarray
    :param y_pred: Binary predictions
    :type y_pred: np.ndarray
    :return: Accuracy, precision, recall and f1 score
    :rtype: tuple
    """
    assert y_true.shape == y_pred.shape, f"y_true and y_pred must have the same shape, but are {y_true.shape} and {y_pred.shape}"

    # Check if y_true and y_pred do not contain any other values than 0 and 1
    assert np.all(np.isin(y_true, [0, 1])), f"y_true must only contain 0 and 1, but contains {np.unique(y_true)}"
    assert np.all(np.isin(y_pred, [0, 1])), f"y_pred must only contain 0 and 1, but contains {np.unique(y_pred)}"

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if TP == 0:
        return 0.5, 0, 0, 0

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1_score


def get_top_n_words(X_data: np.ndarray, vectorizer: CountVectorizer, top_n: int = 5) -> tuple:
    freqs = np.sum(X_data, axis=0)
    top_5_indices = np.argsort(freqs)[::-1][:top_n]  # Order in descending order and get the top 5 indices
    top_5_tokens = vectorizer.get_feature_names_out()[top_5_indices]
    top_5_freqs = freqs[top_5_indices]

    return top_5_tokens, top_5_freqs


if __name__ == "__main__":
    nltk.download("wordnet")
    nltk.download("stopwords")

    fp_data = "./op_spam_v1.4/negative_polarity/"
    max_features = None  # Maximum vocab size
    ngram_range = (1, 1)  # Range of n-grams to include in the vocabulary
    min_df = 0.05  # Minimal document frequency of a word to be included in the vocabulary
    stop_words = set(stopwords.words("english"))
    val_size = 0.2  # Size of the validation set as a percentage of the training set, set to None to disable

    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = pipeline(fp_data, max_features, ngram_range, min_df,
                                                                          stop_words=stop_words, val_size=val_size)

    print(f"X_train: {X_train.shape}; y_train: {len(y_train)}")
    print(f"X_val: {X_val.shape}; y_val: {len(y_val)}")
    print(f"X_test: {X_test.shape}; y_test: {len(y_test)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    print("Top tokens in the vocabulary of the training data:")
    X_data = np.concatenate((X_train, X_val))
    top_n_words, top_n_freqs = get_top_n_words(X_data, vectorizer, top_n=5)
    for token, freq in zip(top_n_words, top_n_freqs):
        print(f"{token} ({freq}x)")

    # Calculate baseline metrics by predicting the majority class
    y_test = np.array(y_test)
    y_pred = np.ones_like(y_test)
    accuracy, precision, recall, f1_score = calculate_metrics(y_test, y_pred)
    print(f"Baseline accuracy: {accuracy:.4f}, precision: {precision:.4f}, recall: {recall:.4f}, f1_score: {f1_score:.4f}")
