import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import data
pd.options.mode.chained_assignment = None

def getCleanTestSentences(sentence):
    if type(sentence) != list:
        sentence = [sentence]
    sentence = [data.cleanSent(sent) for sent in sentence]
    return sentence


def getSentiment(sentences, tokenizer):
    test_data = getCleanTestSentences(sentences)
    tokenizer.fit_on_texts(test_data)
    sequences = tokenizer.texts_to_sequences(test_data)
    X_test = pad_sequences(sequences, padding='post', maxlen=data.MAX_SEQUENCE_LENGTH)
    return X_test


def interpretResult(sentence, result):
    sentiment = {}
    if type(sentence) != list:
        sentence = [sentence]
    for idx in range(len(result)):
        score = np.argmax(result[idx]) - 1
        if score > 0.5:
            sentiment[sentence[idx]] = 'Positive'
        else:
            sentiment[sentence[idx]] = 'Negative'
    return sentiment
