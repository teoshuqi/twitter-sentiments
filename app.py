import numpy as np
from flask import Flask, request, jsonify, render_template, abort
import pickle
import test
import pandas as pd

pd.options.mode.chained_assignment = None
from tensorflow import keras

app = Flask(__name__)
data_file = 'data/playstore_spellchecked.pickle'
token_file = './tokenizer.pickle'
model_file = "./twitter_rnn_200d_0.01.h5"

with open(token_file, 'rb') as fp:
    tokenizer = pickle.load(fp)
model = keras.models.load_model(model_file)


def bad_request(message):
    response = jsonify({'message': message})
    response.status_code = 400
    return response


@app.route('/')
def home():
    return 'App for Sentiment Analysis'


@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON Input
    json = request.get_json(force=True)
    if 'sentences' in json:
        sentences = json['sentences']
    else:
        return bad_request('Wrong Input given. Sentence cannot be found')

    # Get word embeddings for sentence and predict using model
    X_test = test.getSentiment(sentences, tokenizer)
    Y_test = model.predict(X_test)

    # Post process prediction
    result = test.interpretResult(sentences, Y_test)
    return jsonify(result)


@app.route('/results', methods=['POST'])
def results():
    # Get JSON Input
    json = request.get_json(force=True)
    if 'sentences' in json:
        sentences = json['sentences']
    else:
        return bad_request('Wrong Input given. Sentence cannot be found')

    # Get word embeddings for sentence and predict using model
    X_test = test.getSentiment(sentences, tokenizer)
    print(X_test)
    Y_test = model.predict(X_test)
    print(Y_test)
    # Post process results
    results_dict = {}
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        result = Y_test[idx][0]
        positive = round(result*100, 3)
        negative = round(100-positive,3)
        results_dict[sentence] = {'Positive': f'{positive}%' , 'Negative': f'{negative}%'}

    return jsonify(results_dict)


if __name__ == "__main__":
    app.run(debug=True)
