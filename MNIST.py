import tensorflow as tf
from fashion_mnist import img_plot
import numpy as np
from q_gabor import q_gabor
tf.enable_eager_execution()

#tf.keras.utils.generic_utils.get_custom_objects().update({'q_gabor': tf.keras.Activation(q_gabor_activation)})
if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    qqq = [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.5, 1.8]

    for i in range(len(qqq)):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation=q_gabor(qqq[i]).q_gabor_activation),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10)
        ])

        predictions = model(x_train[:1]).numpy()
        print(predictions)

        tf.nn.softmax(predictions).numpy()

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        loss_fn(y_train[:1], predictions).numpy()

        model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])

        history = model.fit(x_train, y_train, epochs=5)

        results = model.evaluate(x_test,  y_test, verbose=2)

        probability_model = tf.keras.Sequential([
          model,
          tf.keras.layers.Softmax()
        ])

        probability_model(x_test[:5])

        img_plot(history, 'MNIST', qqq[i], results[0], results[1])

