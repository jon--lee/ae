import numpy as np
from sparse_network import Network
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import display_images
import time
import sparse
from sparse import Data
import IPython


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

training_epochs = 200
batch_size = 256
display_step = 1
examples_to_show = 10

ap = argparse.ArgumentParser()
ap.add_argument('-l', '--load', required=False, action='store_true')
args = vars(ap.parse_args())
should_load = args['load']

batch_size = 256

net = Network()

if should_load:
    net.load()
else:
    net.init()

for epoch in range(training_epochs):
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch):
        batch_xs, batch_ys = train_data.next_batch(batch_size)
        images, masks = batch_xs[:, :, 0], batch_xs[:, :, 1]
        _, c = net.train({net.X: images, net.masks: masks})
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
            "training loss=", "{:.9f}".format(c))
    if epoch % (display_step * 5) == 0:
        batch_xs, batch_ys = test_data.next_batch(batch_size)
        images, masks = batch_xs[:, :, 0], batch_xs[:, :, 1]
        loss = net.eval({net.X: images, net.masks: masks})
        print("Epoch:", '%04d' % (epoch+1),
            "\ttest loss=", "{:.9f}".format(loss))

display_images.display_sparse(net, test_data, examples_to_show)

IPython.embed()


