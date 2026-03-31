"""
Demo 2 — Pretrained Word Vectors
Session 1: NLP & Word Vectors

Run: python demos/demo2_word2vec.py
Prereqs: pip install gensim

First run downloads the vectors (~66 MB) and caches them locally.
To use the larger Google News vectors (1.6 GB, 300-d), change MODEL_NAME below.
"""

import gensim.downloader as api

# "glove-wiki-gigaword-50" — small & fast, good for classroom demos
# "word2vec-google-news-300" — larger, better results, slower to download
MODEL_NAME = "glove-wiki-gigaword-50"


def load_vectors(model_name: str = MODEL_NAME):
    """Download (first time) and load pretrained word vectors."""
    print(f"Loading '{model_name}' (cached after first download)...")
    return api.load(model_name)


def demo_similarity(wv) -> None:
    """Show word similarity queries."""
    print("\n--- Word Similarity ---")

    pairs = [("cat", "dog"), ("coffee", "tea"), ("king", "queen"), ("cat", "democracy")]
    for a, b in pairs:
        score = wv.similarity(a, b)
        print(f"  similarity('{a}', '{b}') = {score:.3f}")


def demo_most_similar(wv) -> None:
    """Find nearest neighbors for a word."""
    print("\n--- Most Similar Words ---")

    for word in ["coffee", "king", "computer"]:
        print(f"\n  '{word}' →")
        for neighbor, score in wv.most_similar(word, topn=5):
            print(f"    {neighbor:15s} {score:.3f}")


def demo_analogies(wv) -> None:
    """The classic vector arithmetic: A - B + C ≈ ?"""
    print("\n--- Analogies (A - B + C ≈ ?) ---")

    analogies = [
        ("king",   "man",    "woman"),    # → queen
        ("paris",  "france", "germany"),   # → berlin
        ("bigger", "big",    "small"),     # → smaller
    ]
    for pos1, neg, pos2 in analogies:
        results = wv.most_similar(positive=[pos1, pos2], negative=[neg], topn=3)
        top_word, top_score = results[0]
        print(f"  {pos1} - {neg} + {pos2} ≈ {top_word} ({top_score:.3f})")


def demo_odd_one_out(wv) -> None:
    """Which word doesn't belong?"""
    print("\n--- Odd One Out ---")

    groups = [
        ["cat", "dog", "fish", "car"],
        ["coffee", "tea", "juice", "keyboard"],
        ["france", "germany", "italy", "banana"],
    ]
    for group in groups:
        odd = wv.doesnt_match(group)
        print(f"  {group} → '{odd}'")


if __name__ == "__main__":
    wv = load_vectors()

    print(f"\nVocabulary size: {len(wv):,} words")
    print(f"Vector dimensions: {wv.vector_size}")
    print(f"\nVector for 'cat' (first 10 dims): {wv['cat'][:10]}")

    demo_similarity(wv)
    demo_most_similar(wv)
    demo_analogies(wv)
    demo_odd_one_out(wv)
