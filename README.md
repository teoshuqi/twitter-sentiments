# Sentiment Analysis

## Data
Due to limited computational capacity, only part of the sentiment140 data is used to train the models.
<br>
* [sentiment140](http://help.sentiment140.com/home) <br>
* [GloVe](https://nlp.stanford.edu/projects/glove/) Word Embeddings
* [small BERT](https://arxiv.org/abs/1908.08962)


##Installation
Install all required libraries. <br>
Run `data.py` to process and clean data (spell check, word embeddings etc.) <br>
After data and embedding file is generated, run `train.py` to train model and generate pickle file.
```buildoutcfg
pip install requirements.txt
python data.py
python train.py
```


## Reference
* https://github.com/abhinavsagar/machine-learning-deployment
* https://towardsdatascience.com/another-twitter-sentiment-analysis-bb5b01ebad90