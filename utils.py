import glob
import os

import nltk
from nltk.corpus import stopwords

# Punctuation removal
# Stopword removal
# Lemmatization
# CountVectorizer for unigrams and bigrams


def iterate_over_folds(fp_data: str) -> list:
    docs = []

    # Get all txt files using glob
    fps = glob.glob(os.path.join(fp_data, "*", "*.txt"))
    assert len(fps) == 400, f"Expected 400 files, got {len(fps)}"

    for fp in fps:
        with open(fp, "r", encoding="utf-8") as file:
            text = file.read().replace("\n", " ").lower().strip()
        docs.append(text)

    return docs


def load_data(fp_data) -> tuple:
    # Load deceptive
    fp_deceptive = os.path.join(fp_data, "deceptive_from_MTurk")
    docs_deceptive = iterate_over_folds(fp_deceptive)

    # Load truthfull
    fp_truthfull = os.path.join(fp_data, "truthful_from_Web")
    docs_truthfull = iterate_over_folds(fp_truthfull)

    return docs_deceptive, docs_truthfull


if __name__ == "__main__":
    fp_data = "./op_spam_v1.4/negative_polarity/"

    deceptive, truthfull = load_data(fp_data)

    print(deceptive[0])
    print(truthfull[0])
