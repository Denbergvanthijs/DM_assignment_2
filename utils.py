import glob
import os
import string
import unicodedata

import nltk
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
             glob_pattern: str = "*.txt", stop_words: set = set(stopwords.words("english")), unicode_pattern: str = "NFKD",
             val_size: float = 0.2, random_state: int = 42) -> tuple:
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

    # Split train into train and validation, stratified
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train,
                                                      test_size=val_size, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer


if __name__ == "__main__":
    nltk.download("wordnet")
    nltk.download("stopwords")

    fp_data = "./op_spam_v1.4/negative_polarity/"
    max_features = None  # Maximum vocab size
    ngram_range = (1, 1)  # Range of n-grams to include in the vocabulary
    min_df = 0.05  # Minimal document frequency of a word to be included in the vocabulary
    stop_words = set(stopwords.words("english"))

    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer = pipeline(fp_data, max_features, ngram_range, min_df, stop_words=stop_words)

    print(f"X_train: {X_train.shape}; y_train: {len(y_train)}")
    print(f"X_val: {X_val.shape}; y_val: {len(y_val)}")
    print(f"X_test: {X_test.shape}; y_test: {len(y_test)}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
