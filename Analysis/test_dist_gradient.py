import tensorflow as tf
import numpy as np

with tf.device("/cpu:0"):
    x = tf.placeholder(shape=[None,2],dtype=tf.float32)
    y = tf.placeholder(shape=(),dtype=tf.float32)
    W1 = tf.Variable(tf.ones([2,1]))
    W2 = tf.Variable(tf.ones([2,1]))
    b1 = tf.Variable(1.)
    b2 = tf.Variable(1.)
    mu = tf.matmul(x,W1) + b1
    var = tf.nn.softplus(tf.matmul(x,W2) + b2)
    dist = tf.contrib.distributions.Normal(mu,tf.sqrt(var))
    policy = dist.sample(1)
    entropy = tf.multiply(-0.5,tf.nn.bias_add(tf.log(6.28*var),[1]))
    grad = tf.gradients(entropy,W2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    print(sess.run([grad,entropy],feed_dict={x:[[2,2]],y:2}))