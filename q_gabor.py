import tensorflow as tf
import numpy as np
from skimage.filters import gabor

class q_gabor():
    def __init__(self, q, alpha=0.3, f=0.08, theta=0, k=1):
        """"
            :param alpha:
            :param q: opening
            :param f:
            :param theta: angle
            :param k:
        """
        self.q = q
        self.alpha = alpha
        self.f = f
        self.theta = theta
        self.k = k

    @staticmethod
    def sinusoidal_function(f, x):  # s(X) = e^(2*π*f*X*i)
        s = pow(np.e, 2 * np.pi * f * x * 1j)
        return s

    @staticmethod
    def q_exponential_function(x, q):  # w(X) =1/(1+(1-q)*X^2)^(1/(1-q))
        if q == 1:
            w = pow(np.e, -np.pi * pow(x, 2))
        else:
            w = 1 / pow(1 + (1 - q) * x * x, 1 / (1 - q))
        return w

    def q_gabor_1d(self, x, alpha, q, f, theta, k):  # g(X)=k*e^(theta*i)*w(alpha*X)*s(X)
        sinusoidal = self.sinusoidal_function(f, alpha * x)
        q_exponencial = self.q_exponential_function(x, q)
        g = k * pow(np.e, (theta * 1j)) * sinusoidal * q_exponencial
        return g

    @staticmethod
    def q_gabor_2d(x, y, q, k, u, v, p, a, b):
        """"
            :param x = data
            :param y = data
            :param q = opening
            :param k = amplitude
            :param u = X filter frequency
            :param v = Y filter frequency
            :param p = filter phase
            :param a = envelope
            :param b = envelope
        """
        xo = yo = 0
        w = k * (1 / ((1 + (1 - q) * ((a ** 2 * (x - xo) ** 2 + b ** 2 * (y - yo) ** 2))) ** (1 / (1 - q)))) #<- formula diferente!!!!
        s = np.exp((2 * np.pi * (u * x + v * y) + p) * 1j)
        g = w * s
        return g

    def q_gabor_activation(self, x):
        """"
            :param x: data
        """
        x = tf.cast(x, dtype=tf.complex128)
        g = self.q_gabor_1d(x, self.alpha, self.q, self.f, self.theta, self.k)
        return tf.cast(g, dtype=tf.float32)


