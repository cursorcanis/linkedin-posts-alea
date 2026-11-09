"""
What are customers actually complaining about?

A small NLTK project built from the lesson "Lemmatizing and analyzing textual data":
  - WordNetLemmatizer  (nltk.stem.wordnet)
  - pos_tag            (nltk)
  - FreqDist + .plot() (nltk.probability)

The twist: the lesson calls lem.lemmatize(word) with no part of speech, which makes
WordNet treat EVERY word as a noun. So "leaks", "leaked" and "leaking" never collapse
into "leak". Here we feed the pos_tag output into the lemmatizer and compare.

Run:  python analyze_reviews.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # write PNGs instead of opening windows
import matplotlib.pyplot as plt
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer

HERE = Path(__file__).parent

for pkg in ["punkt_tab", "wordnet", "averaged_perceptron_tagger_eng", "stopwords"]:
    nltk.download(pkg, quiet=True)

text = (HERE / "reviews.txt").read_text(encoding="utf-8")

# 1. Tokenize
word_tk = word_tokenize(text.lower())

# 2. Part-of-speech tagging -- on the FULL text, before removing stop words.
#    The tagger needs context. Tagged after filtering, "leaked" and "pour" lose
#    their surrounding words and get mislabeled as non-verbs. (It's still not
#    perfect: "carafe leaks" comes out as JJ + NNS either way.)
tagged_all = pos_tag(word_tk)

# 3. Drop punctuation and stop words (the noise the lesson's plot exposed)
stop = set(stopwords.words("english"))
pos_tagged_words = [(w, t) for w, t in tagged_all if w.isalpha() and w not in stop]
filtered_words = [w for w, _ in pos_tagged_words]
print("POS tags (first 10):", pos_tagged_words[:10], "\n")

# 4. Lemmatize two ways
lem = WordNetLemmatizer()


def to_wordnet_pos(treebank_tag: str) -> str:
    """Map Penn Treebank tags (NN, VBD, JJ...) to WordNet's n / v / a / r."""
    return {"J": "a", "V": "v", "R": "r"}.get(treebank_tag[0], "n")


naive_lemmas = [lem.lemmatize(w) for w in filtered_words]  # as in the lesson
pos_lemmas = [lem.lemmatize(w, to_wordnet_pos(t)) for w, t in pos_tagged_words]

print(f"{'word':<14}{'tag':<6}{'no POS':<14}{'with POS':<12}")
seen = set()
for (w, t), a, b in zip(pos_tagged_words, naive_lemmas, pos_lemmas):
    if a != b and w not in seen:
        seen.add(w)
        print(f"{w:<14}{t:<6}{a:<14}{b:<12}")

fd_naive, fd_pos = FreqDist(naive_lemmas), FreqDist(pos_lemmas)
print(f"\nUnique terms  -> no POS: {fd_naive.B()}  |  with POS: {fd_pos.B()}")
print("'leak' count  -> no POS:", fd_naive["leak"], " |  with POS:", fd_pos["leak"])

# 5. Frequency distributions
fd_raw = FreqDist(word_tk)
print("\nRaw top 5 (noise):     ", fd_raw.most_common(5))
print("Clean lemma top 10:    ", fd_pos.most_common(10))

# Real-world slice: what are people talking ABOUT (nouns) and HOW do they feel (adjectives)?
nouns = FreqDist(l for l, (_, t) in zip(pos_lemmas, pos_tagged_words) if t.startswith("NN"))
adjs = FreqDist(l for l, (_, t) in zip(pos_lemmas, pos_tagged_words) if t.startswith("JJ"))
verbs = FreqDist(l for l, (_, t) in zip(pos_lemmas, pos_tagged_words) if t.startswith("VB"))
print("Top nouns (topics):    ", nouns.most_common(6))
print("Top verbs (what happens):", verbs.most_common(6))
print("Top adjectives (tone): ", adjs.most_common(6))

# Bonus from the lesson: character distribution
fd_alpha = FreqDist(c for c in text.lower() if c.isalpha())
print("Top letters:           ", fd_alpha.most_common(5))

# 6. Plots: the lesson's fd.plot(n, cumulative=False)
for fd, name, title in [
    (fd_raw, "freq_raw.png", "Raw tokens: punctuation and stop words dominate"),
    (fd_naive, "freq_no_pos.png", "Lemmatized WITHOUT POS"),
    (fd_pos, "freq_with_pos.png", "Lemmatized WITH POS"),
]:
    plt.figure(figsize=(10, 4.5))
    fd.plot(15, cumulative=False, title=title, show=False)
    plt.tight_layout()
    plt.savefig(HERE / name, dpi=150)
    plt.close()
print("\nSaved: freq_raw.png, freq_no_pos.png, freq_with_pos.png")
