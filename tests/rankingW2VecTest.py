import tensorflow as tf
import numpy as np
import sys
from aschioppa_tf_layers.negSamplers import NegSampler
from aschioppa_tf_layers.ranking import W2VecRanker

if __name__ == "__main__":
    weights=[1,3,4,8]
    myNeg = NegSampler(10,[1000,1001,1002,1003],weights=weights)
    inputIds = tf.constant(np.array([1000,1001,1001,1002,1003,1003]))
    outputIds = tf.constant(np.array([1003,1002,1001,1002,1000,1000]))
    myLookUp = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer([1000,1001,1002,1003],
                                                            [0,1,2,3],
                                                           key_dtype=tf.int64,
                                                            value_dtype=tf.int64),
        default_value=0)


    myNegLayer = myNeg.getSamplingLayer(inputIds)

    weightsIn = np.array([[-1,-2],[-3,-4],[-5,-6],[-7,-8]])
    weightsOut = 0.1*weightsIn
    myW2Vec = W2VecRanker(4,4,2,weightsIn,weightsOut)
    vecLoss = myW2Vec.sampledLoss(myLookUp.lookup(inputIds),
                                             myLookUp.lookup(outputIds),
                                             myLookUp.lookup(myNegLayer))
    scorePair = myW2Vec.evaluateInPairs(myLookUp.lookup(inputIds),
                                        myLookUp.lookup(outputIds))
    scoreAll = myW2Vec.evaluateOnAll(myLookUp.lookup(inputIds))

    with tf.Session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        print sess.run([inputIds,outputIds])
        print sess.run([vecLoss])
        print sess.run([scorePair])
        print sess.run([scoreAll,tf.shape(scoreAll)])
        print sess.run([tf.log(tf.sigmoid(-scoreAll))])
