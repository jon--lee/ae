import numpy as np
from network import Network
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import display_images
import time
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

import IPython

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
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c = net.train({net.X: batch_xs})
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1),
            "training loss=", "{:.9f}".format(c))
    if epoch % (display_step * 5) == 0:
        batch_xs, batch_ys = mnist.test.next_batch(batch_size)
        loss = net.eval({net.X: batch_xs})
        print("Epoch:", '%04d' % (epoch+1),
            "\ttest loss=", "{:.9f}".format(loss))

display_images.display(net, mnist, examples_to_show)

IPython.embed()