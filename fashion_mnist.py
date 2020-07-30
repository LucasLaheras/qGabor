from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
import numpy as np
from q_gabor import q_gabor
import matplotlib.pyplot as plt


def img_plot(history, base, q, aloss, aacc, val=False):
    history_dict =history.history
    acc = history_dict['acc']
    if val:
        val_acc = history_dict['val_acc']
    loss = history_dict['loss']
    if val:
        val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    # "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
    # b is for "solid blue line"
    if val:
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss q = ' + str(q) + '\nbest '+str(aloss))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_' + base + '_q' + str(q) + '.png')
    plt.show()

    plt.clf()  # clear figure

    plt.plot(epochs, acc, 'r', label='Training acc')
    if val:
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy q = ' + str(q) + '\nbest '+str(aacc))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc_' + base + '_q' + str(q) + '.png')
    plt.show()

if __name__ == '__main__':

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images / 255.0

    test_images = test_images / 255.0

    qqq = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.5, 1.8]

    for i in range(len(qqq)):

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=q_gabor(qqq[i]).q_gabor_activation),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(train_images, train_labels, epochs=10)

        test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
        img_plot(history, 'fashion_mnist', qqq[i], test_loss, test_acc)

        print(qqq)
        print('\nTest accuracy:', test_acc)
        print('Test loss:', test_loss)
