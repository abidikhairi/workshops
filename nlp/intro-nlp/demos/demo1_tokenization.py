"""
Demo 1 — Tokenization & Preprocessing
Session 1: NLP & Word Vectors

Run: python demos/demo1_tokenization.py
Prereqs: pip install nltk
         python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
"""

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def tokenize_and_clean(text: str, language: str = "english") -> list[str]:
    """Tokenize text, remove stop words and punctuation."""
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words(language))
    return [t for t in tokens if t.isalpha() and t not in stop_words]


if __name__ == "__main__":
    text = "The cats are sitting on the mats near the river bank."

    tokens = word_tokenize(text.lower())
    clean = tokenize_and_clean(text)

    print("Original text:", text)
    print("Raw tokens:   ", tokens)
    print("Clean tokens: ", clean)

    # --- Try your own sentence ---
    # custom = "Your sentence here"
    # print("\nCustom clean tokens:", tokenize_and_clean(custom))
