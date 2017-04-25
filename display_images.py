import numpy as np
from network import Network
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import sparse
from sparse import Data
import time


def display(net, mnist, n):
    start = 100
    end = 100 + n
    results = net.run({net.X: mnist.test.images[start:end]})
    f, a = plt.subplots(2, n, figsize=(n, 2))
    for i in range(n):
        actual_image = np.reshape(mnist.test.images[start + i], (28, 28))
        pred_image = np.reshape(results[i], (28, 28))
        a[0][i].imshow(actual_image, cmap='gray')
        a[0][i].set_axis_off()
        a[1][i].imshow(pred_image, cmap='gray')
        a[1][i].set_axis_off()
    plt.show()

def display_sparse(net, data, n):
    start = 100
    end = 100 + n
    images, masks = data.images[start:end, :, 0], data.images[start:end, :, 1]
    results = net.run({net.X: images, net.masks: masks})
    f, a = plt.subplots(2, n, figsize=(n, 2))

    for i in range(n):
        actual_image = np.reshape(images[i] * masks[i], (28, 28))
        pred_image = np.reshape(results[i], (28, 28))
        a[0][i].imshow(actual_image, cmap='gray')
        a[0][i].set_axis_off()
        a[1][i].imshow(pred_image, cmap='gray')
        a[1][i].set_axis_off()
    plt.show()

if __name__ == '__main__':
    net = Network()
    net.load()

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    train_masks = sparse.gen_masks(*mnist.train.images.shape)
    test_masks = sparse.gen_masks(*mnist.test.images.shape)

    n, d = mnist.train.images.shape
    train_images = np.zeros((n, d, 2))
    train_images[:, :, 0], train_images[:, :, 1] = mnist.train.images, train_masks
    train_labels = mnist.train.labels


    n, d = mnist.test.images.shape
    test_images = np.zeros((n, d, 2))
    test_images[:, :, 0], test_images[:, :, 1] = mnist.test.images, test_masks
    test_labels = mnist.test.labels

    train_data = Data(train_images, train_labels)
    test_data = Data(test_images, test_labels)


    display_sparse(net, test_data, 10)
    
    
    

    # net = Network()
    # net.load()
    # mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # display(net, mnist, 10)

