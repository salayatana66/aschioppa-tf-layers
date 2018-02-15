#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import sys
print sys.path

class NegSampler:
    """ keys => numpy array of ids (int64), 
        weights => numpy array of 
                   cumulative rankings for the ids (int64), e.g.
        [1,2,3,4,5] for uniform distro on 5 items
    """
    def __init__(self,num_sampled,keys,weights=None,seed=0):
        if weights is None:
            weights = np.cumsum([1 for _ in xrange(len(keys))])
        # 1st row keys, 2nd probs for sampling
        self.sampleDict = tf.constant(np.array([keys,weights]),dtype=tf.int64)
        self.maxRange = weights[-1]
        self.seed = seed
        self.num_sampled = num_sampled
        
    """ find the first index where the prob exceeds x """
    def firstIndex(self, x):
        return tf.where(self.sampleDict[1,:] >= x)[0]

    """ inputIds => 1-dim tensor of input Ids
        generates tensor of negative ids; collisions are avoided
        by a wrapping trick
        outshape = (inputIds.shape,num_sampled)
    """
    def getSamplingLayer(self, inputIds):
        sampler = tf.random_uniform(shape=[tf.shape(inputIds)[0]*
                                           self.num_sampled],
                                    minval=0, maxval=self.maxRange,
    dtype=tf.int64,
    seed=self.seed)
        sampledIdx = tf.reshape(tf.map_fn(self.firstIndex,sampler),[-1])
        rawSample = tf.reshape(tf.gather(self.sampleDict[0,:],sampledIdx),
                               [-1,self.num_sampled])
        sampledIdx1 = tf.where(tf.reshape(tf.equal(tf.reshape(inputIds,[-1,1]),rawSample),[-1]),
                               sampledIdx+1,sampledIdx)
        sampledIdx2 = tf.where(sampledIdx1 >= tf.to_int64(tf.shape(self.sampleDict)[1])
                                                          ,sampledIdx1-2,sampledIdx1)
        cleanSample = tf.reshape(tf.gather(self.sampleDict[0,:],sampledIdx2),
                               [-1,self.num_sampled])
        return (rawSample,cleanSample)

weights=[1,3,4,8]
test = [1,2,3,4,5,6]
myNeg = NegSampler(10,np.arange(4),weights=weights)
inputIds = tf.constant(np.array([0,0,0,1,2,3,3]))

myNegLayer = myNeg.getSamplingLayer(inputIds)

with tf.Session() as sess:
    print sess.run([inputIds])
    print sess.run([myNegLayer])
    print sess.run(tf.reshape(tf.reshape(test,[2,-1]),[-1]))
    print sess.run(tf.reshape(tf.reshape(test,[-1,2]),[-1]))
