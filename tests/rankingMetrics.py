import tensorflow as tf
import numpy as np
import sys
from aschioppa_tf_layers.ranking import Metrics

if __name__ == "__main__":
    scores = tf.constant(np.array([[0.2, 0.6, 0.8, 1.2],
                                   [0.2, 0.6, 0.8, 1.2],
                                   [0.2, 0.6, 0.8, 1.2],
                                   [0.2, 0.6, 0.8, 1.2]]),dtype=tf.float64)
    trueItems1 = tf.constant(np.array([0, 0, 0, 0]),dtype=tf.int64)
    trueItems2 = tf.constant(np.array([1, 1, 1, 1]),dtype=tf.int64)
    trueItems3 = tf.constant(np.array([2, 2, 2, 2]),dtype=tf.int64)
    trueItems4 = tf.constant(np.array([3, 3, 3, 3]),dtype=tf.int64)
    trueItems5 = tf.constant(np.array([0, 1, 2, 3]),dtype=tf.int64)

    eval1 = Metrics.MeanReciprocalRank(scores, trueItems1, k=4)
    eval2 = Metrics.MeanReciprocalRank(scores, trueItems2, k=4)
    eval3 = Metrics.MeanReciprocalRank(scores, trueItems3, k=4)
    eval4 = Metrics.MeanReciprocalRank(scores, trueItems4, k=4)
    eval5 = Metrics.MeanReciprocalRank(scores, trueItems5, k=4)
    eval6 = Metrics.MeanReciprocalRank(scores, trueItems1, k=3)
    eval7 = Metrics.MeanReciprocalRank(scores, trueItems2, k=3)
    eval8 = Metrics.MeanReciprocalRank(scores, trueItems3, k=3)
    eval9 = Metrics.MeanReciprocalRank(scores, trueItems4, k=3)
    eval10 = Metrics.MeanReciprocalRank(scores, trueItems5, k=3)

    with tf.Session() as sess:
        print sess.run([eval1,eval2,eval3,eval4])
        print sess.run([eval5])
        print sess.run([eval6,eval7,eval8,eval9])
        print sess.run([eval10])


