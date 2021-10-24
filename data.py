import time

tic = time.time()

from nltk.corpus import stopwords
import multiprocessing as mpu
import numpy as np
import pandas as pd
import pickle
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
import warnings
warnings.filterwarnings("ignore")


MAX_NB_WORDS = 40000  # max no. of words for tokenizer
MAX_SEQUENCE_LENGTH = 25  # max length of each entry (sentence), including padding
EMBEDDING_DIM = 200  # embedding dimensions for word vectors (word2vec/GloVe)
GLOVE_DIR = "./glove.6B/glove.6B." + str(EMBEDDING_DIM) + "d.txt"
FILE = 'data/sentiment140_train.csv'
OUTPUT_FILE = 'data/tweets.pickle'
OUTPUT_FILE_BERT = 'data/tweets_bert.pickle'


def check_row(a):
    if a % 40 == 1:
        return False
    return True


def readTwitterData(excel_file):
    twitter_data = pd.read_csv(excel_file, encoding_errors='ignore', header=None, encoding="utf-8",
                               skiprows=lambda x: check_row(x)).iloc[:, [-1, 0]]
    twitter_data.columns = ['Text', 'Sentiment']

    twitter_data['Text'] = twitter_data['Text'].apply(cleanSent)
    tweets = np.array(twitter_data['Text'])
    labels = np.array(twitter_data['Sentiment'])
    labels[labels > 0] = 1
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    labels = to_categorical(labels)
    return tweets, labels


def cleanSent(sentence):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    sentence = re.sub("@[A-Za-z0-9_]+|#[A-Za-z0-9_]+|http\S+|www.\S+", "", sentence) #url,hashtags and mentions
    sentence = BeautifulSoup(sentence, 'lxml').get_text() # html
    sentence = re.sub(r'[^a-zA-Z ]+', ' ', sentence) # remove punctuations, numbers etc.
    arr = []
    for w in sentence.split():
        if w not in stop_words:
            arr.append(lemmatizer.lemmatize(w.lower()))
    clean_sentence = ' '.join(arr)
    return clean_sentence


if __name__ == '__main__':
    tweets, labels = readTwitterData(FILE)
    print(tweets.shape)

    with open(OUTPUT_FILE_BERT, 'wb') as fp:
        pickle.dump((tweets, labels), fp)

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    print('Vocabulary size:', len(word_index))

    data = pad_sequences(sequences, padding='post', maxlen=MAX_SEQUENCE_LENGTH)

    ## shuffle data
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    ## map to glove
    embeddings_index = {}
    f = open(GLOVE_DIR, 'r', encoding="utf8")
    print('Loading GloVe from:', GLOVE_DIR, '...', end='')
    for line in f:
        values = line.split()
        word = values[0]
        embeddings_index[word] = np.asarray(values[1:], dtype='float32')
    f.close()
    print("Done.\n Proceeding with Embedding Matrix...", end="")

    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print(" Completed!")

    with open(OUTPUT_FILE, 'wb') as fp:
        pickle.dump((data, labels, embedding_matrix, len(word_index)), fp)

    # saving
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    toc = time.time()
    print('Done in {:.4f} mins'.format((toc - tic) / 60))
