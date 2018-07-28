import tensorflow as tf
import numpy as np

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
            weights = np.cumsum([1 for _ in range(len(keys))])
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


class ShardedNegUniformSampler:
    def __init__(self,num_sampled,seed=0):
        self.seed = seed
        self.num_sampled = num_sampled

    def getSamplingLayer(self, inputIds, minIds, maxIds):
        # instantiate a uniform sampler in [0,1)
        # (Batch, num_sampled)
        sampler = tf.random_uniform(shape=[tf.shape(inputIds)[0],
                                           self.num_sampled],
                                    minval=0, maxval=1,
    dtype=tf.float32,
    seed=self.seed)
        # create the rescaling width; need to add 1.0 as sampler
        # in [0,1)
        width = tf.to_float(maxIds)-tf.to_float(minIds)+tf.constant(1.0)
        # reshape and tile to combine with sampler
        widthRes = tf.tile(tf.reshape(width,[-1,1]),
                           [1,self.num_sampled])
        # reshape and tile the minimum
        minRes = tf.tile(tf.reshape(tf.to_float(minIds),[-1,1]),
                           [1,self.num_sampled])
        # take the ceiling
        floor = tf.to_int64(tf.floor(sampler*widthRes + minRes))
        # take positive collisions
        posColl = tf.equal(floor,tf.reshape(inputIds,[-1,1]))
        # go to the next element
        nextColl = tf.where(posColl,floor+tf.constant(1,dtype=tf.int64),floor)
        # if going to the next moves above maximum
        aboveColl = nextColl > tf.reshape(maxIds,[-1,1])
        # then resolve going back to min
        resColl = tf.where(aboveColl,tf.to_int64(minRes),nextColl)
        return resColl

    

