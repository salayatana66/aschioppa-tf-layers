import tensorflow as tf
import numpy as np
import sys
print(sys.path)
from aschioppa_tf_layers.negSamplers import ShardedNegUniformSampler as Snu

if __name__ == "__main__":
    inputIds = tf.constant(np.array([3,1,2,3,10,11,23]))
    minIds = tf.constant(np.array([3,0,0,2,8,9,15]))
    maxIds = tf.constant(np.array([3,5,3,3,10,15,40]))
    myNeg = Snu(10)
    myNegLayer = myNeg.getSamplingLayer(inputIds,minIds,maxIds)

    with tf.Session() as sess:
        print(sess.run([inputIds]))
        print(sess.run(tf.shape(inputIds)))
        print(sess.run([minIds]))
        print(sess.run([maxIds]))
        print(sess.run([myNegLayer]))
        #print(sess.run([tf.shape(myNegLayer)]))


