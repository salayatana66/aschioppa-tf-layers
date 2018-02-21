import tensorflow as tf
import numpy as np
import sys
print sys.path
from aschioppa_tf_layers.negSamplers import NegSampler

if __name__ == "__main__":
    weights=[1,3,4,8]
    test = [1,2,3,4,5,6]
    myNeg = NegSampler(10,np.arange(4),weights=weights)
    inputIds = tf.constant(np.array([0,0,0,1,2,3,3]))

    myNegLayer = myNeg.getSamplingLayer(inputIds)

    with tf.Session() as sess:
        print sess.run([inputIds])
        print sess.run([myNegLayer])
