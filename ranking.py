import tensorflow as tf
import numpy as np

"""
 Class of a word-to-vec model which uses matrices U & V with latentfactors to recommend;
 U[i,:].dot(V[j,:]) is the score of user/item i and item/user j
"""

class W2VecRanker:
    """
    num_items_in => number of observed item / user
    num_items_out => number of item / users to recommend
    Note: weights can be preloaded but there is not check for the right shape at the moment
    """
    def __init__(self,num_items_in, num_items_out, num_latent_factors,
                  prior_in_weights=None, prior_out_weights=None):
        self.num_items_in = num_items_in
        self.num_items_out = num_items_out
        self.num_latent_factors = num_latent_factors
        if prior_in_weights is None:
            self.weights_in = tf.get_variable(name="weights_in",shape = [self.num_items_in,
                                                       self.num_latent_factors],
                                              initializer =
                                              tf.truncated_normal_initializer(
                                                  mean = 0.0, stddev=1.0/np.sqrt(self.num_latent_factors),dtype=tf.float64),dtype=tf.float64)
        else:
            self.weights_in = tf.get_variable(name="weights_in",
                                              initializer =
                                              tf.constant(value=prior_in_weights,dtype=tf.float64),dtype=tf.float64)
        if prior_out_weights is None:
            self.weights_out = tf.get_variable(name="weights_out",shape=[self.num_items_out,
                                                      self.num_latent_factors],
                                               initializer =
                                              tf.truncated_normal_initializer(
                                                  mean = 0.0, stddev=1.0/np.sqrt(self.num_latent_factors),dtype=tf.float64),dtype=tf.float64)
        else:
            self.weights_out = tf.get_variable(name="weights_out",
                                               initializer =
                                               tf.constant(value=prior_out_weights,dtype=tf.float64),dtype=tf.float64)

    """
    helper function returns weights for U (In)
    """
    def returnInSlice(self,inputTensor):
        return tf.gather(self.weights_in,inputTensor)

    """
    helper function returns weights for V (Out)
    """
    def returnOutSlice(self,inputTensor):
        return tf.gather(self.weights_in,inputTensor)

    """
    Loss trained like in Work to Vec
    inputTensor => seen items/users
    trueOutput => positive examples
    negSampledOutput => negative (say via a sampler)
    """
    def sampledLoss(self,inputTensor,trueOutput,negSampledOutput):
        U = self.returnInSlice(inputTensor)
        Vtrue = self.returnOutSlice(trueOutput)
        Vsampled = self.returnOutSlice(negSampledOutput)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])

        positive = tf.log(tf.sigmoid(tf.map_fn(lambda x: tf.reduce_sum(U[x,:]*Vtrue[x,:]),
                             idxRange,dtype=tf.float64)))
        negative = tf.reduce_sum(
            tf.log(
                tf.sigmoid(-tf.map_fn(lambda x: tf.tensordot(U[x,:],Vsampled[x,:,:],
                                                    axes=[[0],[1]]),
                                      idxRange,dtype=tf.float64))),
            axis=1)
        loss = -tf.reduce_sum(positive+negative)
        return loss

    """
    Evaluates items pairwise
    """
    def evaluateInPairs(self, inputTensor, outputTensor):
        U = self.returnInSlice(inputTensor)
        V = self.returnOutSlice(outputTensor)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])
        score = tf.map_fn(lambda x: tf.reduce_sum(U[x,:]*V[x,:]),
                             idxRange,dtype=tf.float64)
        return score

    """
    Evaluates an input item against all the output one
    """
    def evaluateOnAll(self, inputTensor):
        U = self.returnInSlice(inputTensor)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])
        score = tf.map_fn(lambda x: tf.tensordot(U[x,:],self.weights_out,
                                                  axes=[[0],[1]]),
                             idxRange,dtype=tf.float64)
        return score


"""
Metrics to evaluate Ranking Algorithms
"""

class Metrics:
    """
    Mean Reciprocal Rank where in each batch there is only one selected item
    """
    @staticmethod
    def MeanReciprocalRank(scores, trueItems, k):
        # TF has no argsort: rank values descending, find indices
        sValues, sIndices = tf.nn.top_k(scores,k=k)
        # to iterate
        idxRange = tf.range(start=0,limit=tf.shape(trueItems)[0])
        # find the indices where there is a match
        matchedQ = tf.where(
            tf.map_fn(lambda x:
                      tf.size(tf.where(tf.equal(sIndices[x,:],tf.to_int32(trueItems[x])))),
                      idxRange)>0)
        # compute the ranks of the items for which there was a match
        computedRanksQ = tf.to_double(tf.map_fn(lambda x:
                                         tf.where(tf.equal(sIndices[x,:],
                                                                   tf.to_int32(trueItems[x])))[0],
                                         tf.reshape(matchedQ,[-1])))


        # average the reciprocal ranks;
        # as some items might be missing you need to take sum and divide by the shape
        return tf.reduce_sum(1.0/(computedRanksQ+1.0))/tf.to_double(tf.shape(idxRange)[0])

