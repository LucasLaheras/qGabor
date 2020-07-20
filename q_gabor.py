import tensorflow as tf
import numpy as np

eps = np.finfo(float).eps


# s(x)
def sinusoidal_calc(f, x):
    s = np.exp((2 * np.pi * f * x) * 1j)
    return s


# w(x)
def q_exponential_calc(x, q):
    base = 1 + (1 - q) * pow(x, 2)
    w = 1 / pow(base, 1 / (1 - q))
    return w


def q_gabor_1d(x, alpha, q, f, theta, k):
    """""
        x = data
        alpha = 
        q = opening
        f = 
        theta = angle
        k = 
    """""

    sinusoidal = sinusoidal_calc(x, f)
    q_exponencial = q_exponential_calc(alpha * x, q)
    potentiation = pow(eps, theta * 1j).real
    g = k * potentiation * sinusoidal * q_exponencial
    return g


def q_gabor_2d(x, y, q, k, u, v, p, a, b):
    """""
        x and y = data
        q = opening
        k = amplitude
        u and v = filter frequency
        p = filter phase
        a and b = envelope 
    """""

    xo = yo = 0
    w = k * (1 / ((1 + (1 - q) * ((a ** 2 * (x - xo) ** 2 + b ** 2 * (y - yo) ** 2))) ** (1 / (1 - q)))) #<- formula diferente!!!!
    s = np.exp((2 * np.pi * (u * x + v * y) + p) * 1j)
    g = w * s
    return g
