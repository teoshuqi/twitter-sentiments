import os
import shutil

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import mixed_precision
from sklearn.metrics import confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# restrict precision
mixed_precision.set_global_policy('mixed_float16')


tfhub_handle_preprocess = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
tfhub_handle_encoder = "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/2"
MODEL_FILE = "s140_bert.h5"
DATA_FILE = 'data/tweets_bert.pickle'

with open(DATA_FILE, 'rb') as fp:
    (X, Y) = pickle.load(fp)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=56)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=56)

if os.path.isfile(MODEL_FILE):
    model = tf.keras.models.load_model(MODEL_FILE, custom_objects={'KerasLayer':hub.KerasLayer})
else:
    ## Input and Preprocessing
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)

    ## BERT Sentence Embedding
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    output = outputs['pooled_output']  ## extracts the output from the first token/cls token
    # ## Classification head
    class_output = tf.keras.layers.Dense(2, activation=tf.keras.activations.softmax,
                                         kernel_initializer=tf.keras.initializers.GlorotNormal(),
                                         name='classifier')(output)
    model = tf.keras.Model(inputs=text_input, outputs=class_output)

initial_learning_rate = 0.000001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=False)
adam = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=32, epochs=1, validation_data=(X_val, y_val), )
tf.keras.models.save_model(model, MODEL_FILE, save_format="h5")

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
fig.savefig(MODEL_FILE[:-3]+'_train4.png')

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
figure.savefig(MODEL_FILE[:-3]+'.png', dpi=400)