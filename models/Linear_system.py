# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:25:07 2017

@author: franc
"""

import collections

import tensorflow as tf
import numpy as np

def random_linear_observations(n_obs,x,noise):
    
    xl = np.shape(x)
    A = tf.random_uniform([n_obs,xl[0]])
    er = tf.random_normal([n_obs,1], 0, 1, dtype=tf.float32)
    b = tf.matmul(A,x)+noise*er

    return A, b

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
xt, _ = mnist.test.next_batch(1)
x = np.transpose(xt) # vector being observed (our image)
sh = np.shape(x)

no = 20 # number of observations
noise = 0.01

A, b = random_linear_observations(no,x,noise)