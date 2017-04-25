import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import IPython

class Network():

    learning_rate = 0.01
    training_epochs = 20
    batch_size = 256

    n_hidden_1 = 256 # 1st layer num features
    n_hidden_2 = 128 # 2nd layer num features
    n_input = 784 # MNIST data input (img shape: 28*28)



    def __init__(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder("float", [None, self.n_input])
            
            self.weights = {
                'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
                'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
                'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
                'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),
            }
            self.biases = {
                'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
                'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
                'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),
            }

            self.enc = self.encoder(self.X)
            self.dec = self.decoder(self.enc)

            self.output = self.dec
            self.loss = tf.reduce_mean(tf.pow(self.output - self.X, 2))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)



    def encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer2

    def decoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer2


    def init(self):
        with self.graph.as_default():
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        return self.sess

    def save(self):
        with self.graph.as_default():
            with self.sess.as_default():
                saver = tf.train.Saver()
                save_path = saver.save(self.sess, "./tmp/model.ckpt")
                print("Model saved in file: %s" % save_path)

    def load(self):
        with self.graph.as_default():
            self.sess = tf.Session()
            with self.sess.as_default():
                saver = tf.train.Saver()
                saver.restore(self.sess, "./tmp/model.ckpt")
                print("Model restored.")

    def run(self, fd):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.sess.run(self.output, fd)

    def eval(self, fd):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.sess.run(self.loss, fd)

    def train(self, fd):
        with self.graph.as_default():
            with self.sess.as_default():
                return self.sess.run([self.optimizer, self.loss], feed_dict=fd)


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    net = Network()
    net.init()
    output = net.run({net.X: mnist.test.images[:1]})
    print output[0, :10]
    net.save()

    net2 = Network()
    net2.init()
    output = net2.run({net2.X: mnist.test.images[:1]})
    print output[0, :10]







