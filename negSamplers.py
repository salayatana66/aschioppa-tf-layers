import tensorflow as tf
import numpy as np
import sys
print sys.path

# Implements a negative sampler layers
# The keys are the items to be sampled, the weights are used
# to generate a distribution on the items (uniform by default)

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
        # the maximal value for weights
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
        # instantiate a uniform sampler
        sampler = tf.random_uniform(shape=[tf.shape(inputIds)[0]*
                                           self.num_sampled],
                                    minval=0, maxval=self.maxRange,
    dtype=tf.int64,
    seed=self.seed)
        # pull out the sampled indices
        sampledIdx = tf.reshape(tf.map_fn(self.firstIndex,sampler),[-1])
        # pull out the sampled values and reshape to [inputSize,num_sampled]
        rawSample = tf.reshape(tf.gather(self.sampleDict[0,:],sampledIdx),
                               [-1,self.num_sampled])
        # These operations resolve collisions by going to the next value or
        # the previous one
        sampledIdx1 = tf.where(tf.reshape(tf.equal(tf.reshape(inputIds,[-1,1]),
                                                   rawSample),[-1]),
                               sampledIdx+1,sampledIdx)
        sampledIdx2 = tf.where(sampledIdx1 >= tf.to_int64(tf.shape(self.sampleDict)[1])
                                                          ,sampledIdx1-2,sampledIdx1)
        # subset to get the sample without collisions
        cleanSample = tf.reshape(tf.gather(self.sampleDict[0,:],sampledIdx2),
                               [-1,self.num_sampled])

        return cleanSample


if __name__ == "__main__":
    weights=[1,3,4,8]
    test = [1,2,3,4,5,6]
    myNeg = NegSampler(10,np.arange(4),weights=weights)
    inputIds = tf.constant(np.array([0,0,0,1,2,3,3]))

    myNegLayer = myNeg.getSamplingLayer(inputIds)

    with tf.Session() as sess:
        print sess.run([inputIds])
        print sess.run([myNegLayer])


