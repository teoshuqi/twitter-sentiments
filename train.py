import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os.path
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

plt.style.use('ggplot')

data_file = 'data/tweets.pickle' #'data/playstore_spellchecked.pickle'
embedding_file = './glove_embeddings.pickle'
model_file = "./s140_200d_0.001.h5"
test = 0.05
valid = 0.05
seed = 56
epochs = 100

with open(data_file, 'rb') as fp:
    (X, Y, embedding_matrix, max_word_index) = pickle.load(fp)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test, random_state=seed)
print(X_train.shape)
if os.path.isfile(model_file):
    model = keras.models.load_model(model_file)
else:

    print('Prepare Model')
    model = keras.Sequential()  ### Input X dim = (N, Timesteps, embedding dim)

    # Add an Embedding layer expecting input vocab of size max_word_index, and output embedding dimension of size .
    model.add(layers.Embedding(input_dim=max_word_index+1, output_dim=embedding_matrix.shape[1], input_length=len(X_train[0]),
                               weights=[embedding_matrix], trainable=False))

    # LSTM layer
    model.add(layers.Bidirectional(layers.LSTM(16, return_sequences=True,
                                               kernel_regularizer=tf.keras.regularizers.L2(0.0001))))
    model.add(layers.Bidirectional(layers.LSTM(16, kernel_regularizer=tf.keras.regularizers.L2(0.0001))))

    # Dense layer
    model.add(layers.Dense(16, activation=tf.keras.activations.relu,
                           kernel_initializer=tf.keras.initializers.GlorotNormal(),
                           kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(layers.Dense(2, activation=tf.keras.activations.softmax))

    model.summary()

print('Train Model')
initial_learning_rate = 0.00004
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5000,
    decay_rate=0.96,
    staircase=False)
adam = keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, np.array(y_train), batch_size=128, epochs=100, validation_split=valid)

model.save(model_file)

# Training Performance
history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

epochs = range(1, len(acc) + 1)
fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.subplot(1, 1, 1)
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
fig.savefig(model_file[:-3]+'_train.png')

# Test Performance
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_test = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
class_names = ['Negative', 'Positive']

ax = sns.heatmap(cm, annot=True, fmt = 'g', cmap='Blues')  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title(f'Accuracy: {accuracy*100}%', fontsize = 10 )
ax.xaxis.set_ticklabels(class_names)
ax.yaxis.set_ticklabels(class_names)
figure = ax.get_figure()
figure.savefig(model_file[:-3]+'.png', dpi=400)