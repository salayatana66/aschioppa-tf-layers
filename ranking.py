import tensorflow as tf
import numpy as np

"""Represents a word-to-vec which uses matrices U & V with latentfactor to recommend;
 U[i,:].dot(V[j,:]) is the score of user/item i and item/user j"""
class W2VecRanker:
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
                                                  mean = 0.0, stddev=1.0/np.sqrt(self.num_latent_factors),dtype=tf.float64),type=tf.float64)
        else:
            self.weights_in = tf.get_variable(name="weights_in",
                                              initializer =
                                              tf.constant(value=prior_in_weights,dtype=tf.float64),dtype=tf.float64)
        if prior_out_weights is None:
            self.weights_out = tf.get_variable(name="weights_out",shape=[self.num_items_out,
                                                      self.num_latent_factors],
                                               initializer =
                                              tf.truncated_normal_initializer(
                                                  mean = 0.0, stddev=1.0/np.sqrt(self.num_latent_factors),dtype=tf.float64),type=tf.float64)
        else:
            self.weights_out = tf.get_variable(name="weights_out",
                                               initializer =
                                               tf.constant(value=prior_out_weights,dtype=tf.float64),dtype=tf.float64)

    def returnInSlice(self,inputTensor):
        return tf.gather(self.weights_in,inputTensor)

    def returnOutSlice(self,inputTensor):
        return tf.gather(self.weights_in,inputTensor)

    def sampledLoss(self,inputTensor,trueOutput,negSampledOutput):
        U = self.returnInSlice(inputTensor)
        Vtrue = self.returnOutSlice(trueOutput)
        Vsampled = self.returnOutSlice(negSampledOutput)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])

        positive = tf.map_fn(lambda x: tf.reduce_sum(U[x,:]*Vtrue[x,:]),
                             idxRange,dtype=tf.float64)
        negative = tf.reduce_sum(
            tf.log(
                tf.sigmoid(-tf.map_fn(lambda x: tf.tensordot(U[x,:],Vsampled[x,:,:],
                                                    axes=[[0],[1]]),
                                      idxRange,dtype=tf.float64))),
            axis=1)
        loss = -tf.reduce_sum(positive+negative)
        return loss

    def evaluateInPairs(self, inputTensor, outputTensor):
        U = self.returnInSlice(inputTensor)
        V = self.returnOutSlice(outputTensor)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])
        score = tf.map_fn(lambda x: tf.reduce_sum(U[x,:]*V[x,:]),
                             idxRange,dtype=tf.float64)
        return score

    def evaluateOnAll(self, inputTensor):
        U = self.returnInSlice(inputTensor)
        idxRange = tf.range(start=0,limit=tf.shape(U)[0])
        score = tf.map_fn(lambda x: tf.tensordot(U[x,:],self.weights_out,
                                                  axes=[[0],[1]]),
                             idxRange,dtype=tf.float64)
        return score

