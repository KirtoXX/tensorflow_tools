import tensorlayer as tl
import tensorflow as tf


def my_network(image):

    network = tl.layers.InputLayer(image)
    network = tl.layers.DenseLayer(network,n_units=100,act=tf.nn.relu)
    network = tl.layers.DenseLayer(network,n_units=100,act=tf.nn.relu)
    network = tl.layers.DenseLayer(network,n_units=10,act=tf.nn.softmax)

    return network.outputs
