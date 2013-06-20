# sentiment-classifier.py
# classifies the sentiment of a document
# uses Movie Review corpus from Cornell

import nltk
from nltk.corpus import movie_reviews


all_words = nltk.FreqDist(word.lower() for word in movie_reviews.words())
word_features = all_words.keys()[:2000]


def document_features(document):
    document_vocab = set(document)
    features = {}

    for word in word_features:
        features["contains(%s)" % word] = (word in document_vocab)

    return features