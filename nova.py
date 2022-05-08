import numpy
import tensorflow
import tflearn
import random
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

nltk.download("averaged_perceptron_tagger")
stemmer = LancasterStemmer()

with open("model.json") as file:
    data = json.load(file)
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            _words = nltk.word_tokenize(pattern)
            words.extend(_words)
            docs_x.append(_words)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    print(words)
    print(labels)
    print(docs_x)
    print(docs_y)
