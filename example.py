import tensorflow as tf
import os
import numpy as np
import cv2
from transform import  *

def imshow(out):
    out = np.array(out * 255, dtype=np.uint8)
    cv2.imshow('img3', out[0])
    cv2.waitKey()
    cv2.imshow('img4', out[1])
    cv2.waitKey()

def get_theta():
    M = np.array([[1, 0, 0], [0, 1, 0]])
    #M = np.array([[0.707, -0.707, 0.], [0.707, 0.707, 0.]])
    M = np.resize(M, (2, 2, 3))
    M = np.reshape(M, (2, 6))
    return M

def get_batch():
    DIMS = (400, 400)
    CAT1 = 'cat1.jpg'
    CAT2 = 'cat2.jpg'

    img1 = cv2.imread(CAT1)
    img2 = cv2.imread(CAT2)

    img1 = np.array(cv2.resize(img1, DIMS)) / 255.0
    img2 = np.array(cv2.resize(img2, DIMS)) / 255.0

    t_img1 = np.array(img1 * 255.0, dtype=np.uint8)
    t_img2 = np.array(img2 * 255.0, dtype=np.uint8)
    cv2.imshow('img1', t_img1)
    cv2.waitKey()
    cv2.imshow('img2', t_img2)
    cv2.waitKey()

    img1 = np.reshape(img1, (1, 400, 400, 3))
    img2 = np.reshape(img2, (1, 400, 400, 3))
    img = np.concatenate((img1, img2), axis=0)
    return img

x = tf.placeholder(tf.float32, [None, 400, 400, 3])
n_fc = 6

initial = np.array([[0.5, 0, 0],[0, 0.5, 0]]).astype('float32')
initial = initial.flatten()

w_fc1 = tf.get_variable('W_fc1', [400*400*3, n_fc], initializer=tf.constant_initializer(0.0))
b = tf.get_variable('b', initializer=initial)
x_ = tf.reshape(x, [-1, 400*400*3])
theta = tf.matmul(x_, w_fc1) + b
x_tran = transform(theta, x, [2, 200, 200, 3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
img = get_batch()
output = sess.run(x_tran, feed_dict={x: img})
imshow(output)
