from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from q_gabor import q_gabor
import matplotlib.pyplot as plt
from fashion_mnist import img_plot

import numpy as np
from sklearn.metrics import roc_curve

from sklearn.metrics import auc

from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    imdb = keras.datasets.imdb

    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


    # Um dicionário mapeando palavras em índices inteiros
    word_index = imdb.get_word_index()

    # Os primeiros índices são reservados
    word_index = {k: (v+3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2  # unknown
    word_index["<UNUSED>"] = 3

    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    print(decode_review(train_data[0]))

    qqq = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.5, 1.8]

    for i in range(len(qqq)):
        decode_review(train_data[0])
        train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                                value=word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)

        test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                               value=word_index["<PAD>"],
                                                               padding='post',
                                                               maxlen=256)

        # O formato de entrada é a contagem vocabulário usados pelas avaliações dos filmes (10000 palavras)
        vocab_size = 10000

        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=q_gabor(qqq[i]).q_gabor_activation))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        x_val = train_data[:10000]
        partial_x_train = train_data[10000:]

        y_val = train_labels[:10000]
        partial_y_train = train_labels[10000:]


        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=40,
                            batch_size=512,
                            validation_data=(x_val, y_val),
                            verbose=1)

        results = model.evaluate(test_data,  test_labels, verbose=2)
        img_plot(history, 'IMDB', qqq[i], results[0], results[1], True)

