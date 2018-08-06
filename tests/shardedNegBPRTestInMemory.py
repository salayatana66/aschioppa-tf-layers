import tensorflow as tf
import numpy as np
import sys
from aschioppa_tf_layers.negSamplers import ShardedNegUniformSampler as Snu
from aschioppa_tf_layers.ranking import SimpleFactorRanker as Sfr
import time
import argparse

parser = argparse.ArgumentParser(description='Test The BPR with on the fly negative samples')
parser.add_argument('--users', dest = 'users',type=int, nargs=1,required=True)
parser.add_argument('--items', dest = 'items',type=int,nargs=1,required=True)
parser.add_argument('--examples', dest = 'examples', type = int, nargs=1,required=True)
parser.add_argument('--batch_size', dest = 'batch_size', type = int,nargs=1,required=True)
parser.add_argument('--num_negs', dest = 'num_negs', type = int,nargs=1,required=True)
parser.add_argument('--num_latent', dest = 'num_latent', type = int,nargs=1,required=True)
args = parser.parse_args()

if __name__ == "__main__":
    t0 = time.time()

    np.random.seed(12)
    # sample users
    users=np.reshape(np.random.random_integers(low = 0, high = args.users[0],
                                                 size = args.examples[0]),[-1,1])
    # sample items
    items = np.sort(np.reshape(np.random.random_integers(low = 0, high = args.items[0],
                                                         size = args.examples[0]*3),[-1,3]),axis=1)

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def gen():
        for i in range(0,args.examples[0],args.batch_size[0]):
            imax = min(i+args.batch_size[0],args.examples[0])
            yield {'user' : users[i:imax,0],
                                              'item' : items[i:imax,1],
                                              'minItem' : items[i:imax,0],
                                              'maxItem' : items[i:imax,2]}

    myDataset = tf.data.Dataset.from_generator(gen,output_types={'user': tf.int64,
                                                    'item' : tf.int64,
                                                    'minItem' : tf.int64,
                                                                 'maxItem' : tf.int64},
                                               output_shapes = {'user' : tf.TensorShape([None]),
                                                                'item': tf.TensorShape([None]),
                                                                'minItem' : tf.TensorShape([None]),

                                                                'maxItem' : tf.TensorShape([None])})
    # for this we cannot batch
    #    myDataset = myDataset.batch(1)
    #myDataset = myDataset.batch(args.batch_size[0])
    #myDataset = myDataset.prefetch(args.batch_size[0]*5)
    iterator = myDataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    myNeg = Snu(args.num_negs[0])
    myNegLayer = myNeg.getSamplingLayer(next_element['item'],next_element['minItem'],
                                                     next_element['maxItem'])

    myRanker = Sfr(args.items[0]+1,args.users[0]+1,args.num_latent[0])
    itemScore,negScore=myRanker.getFactorProducts(next_element['user'],next_element['item'],myNegLayer)
    loss = myRanker.getBPRLoss(itemScore,negScore)


    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss,
                                                global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = time.time()
    for ii in range(args.examples[0]):
        try:
            sess.run([optimizer])
        except tf.errors.OutOfRangeError:
            break
    print('End benchmark', time.time()-t0)






