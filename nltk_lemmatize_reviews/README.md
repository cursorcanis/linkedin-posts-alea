# What are customers actually complaining about? (NLTK)

A small project built from the LinkedIn Learning lesson **"Lemmatizing and analyzing textual data"**
(*Python for Data Science and Machine Learning*).

## What the lesson covered
| Tool | Import | Purpose |
|---|---|---|
| `WordNetLemmatizer` | `from nltk.stem.wordnet import WordNetLemmatizer` | Reduce words to their dictionary form (lemma) |
| `pos_tag` | `from nltk import pos_tag` | Tag each token with a part of speech (NN, VBD, JJ, ...) |
| `FreqDist` | `from nltk.probability import FreqDist` | Count tokens, then `fd.plot(30, cumulative=False)` |

## The twist in this project
In the lesson, `lem.lemmatize(word)` is called **without a part of speech**. WordNet then treats every
word as a noun, so `leaks / leaked / leaking` stay three separate words and the complaint gets split up.

This project passes the `pos_tag` output into the lemmatizer (`lemmatize(word, pos)`). Then the
nouns, verbs and adjectives are counted separately, which turns a word count into a rough
summary of the reviews: what people mention, what goes wrong, and how they describe it.

## Run
```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python analyze_reviews.py
```
Outputs: a console report plus `freq_raw.png`, `freq_no_pos.png` and `freq_with_pos.png`.

`reviews.txt` holds 12 made-up coffee-maker reviews. Replace it with your own text to analyze something else.
