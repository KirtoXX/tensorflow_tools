import tensorflow as tf
from mutul_device import *
from network import my_network

with tf.Graph().as_default():

    X = tf.placeholder(dtype=tf.float32,shape=[None,100])
    y = tf.placeholder(dtype=tf.float32,shape=[None,10])

    mutul_gpu1 = mutul_gpu()
    mutul_gpu1.set_device(['/gpu:0','/cpu:0'])  #此处必须传入一个list
    mutul_gpu1.set_paramater(input=X,
                            real_target=y,
                            network=my_network,
                            loss=tf.losses.softmax_cross_entropy,
                            train_op=tf.train.AdadeltaOptimizer)

    train_op = mutul_gpu1.get_trainop()

