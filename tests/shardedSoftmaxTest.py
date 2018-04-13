import tensorflow as tf
import numpy as np
import sys
from aschioppa_tf_layers.shardedSoftmax import ShardedSoftmax

if __name__ == "__main__":
    sf = ShardedSoftmax(3,5)
    W = np.array([[4.0,3.2,1.0,0.5,2.17],[-1.0,2.0,1.5,7.4,3.25],[-3.5,2.17,4.7,3.2,1.33]])
    b = np.array([[-0.25],[-2.3],[-15.4],[-17.3],[124.2]])
    weights = sf.getWeights(initializers={'W':tf.constant(W,dtype=tf.float32), 'b' : tf.constant(b,dtype=tf.float32)})
    lower = np.array([0,3])
    upper = np.array([2,4])
    I = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]])
    inputs = tf.constant(I,dtype=tf.float32)
    lb = tf.constant(lower)
    ub = tf.constant(upper)
    
    out = sf.getLayer(inputs,weights,lb,ub)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(weights))
        print(sess.run(out))
        


        
