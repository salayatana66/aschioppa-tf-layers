import tensorflow as tf
import numpy as np
import sys
from aschioppa_tf_layers.negSamplers import ShardedNegUniformSampler as Snu
from aschioppa_tf_layers.ranking import SimpleFactorRanker as Sfr
import time
import argparse

class ConvertToInt(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
         if nargs is not None:
             raise ValueError("nargs not allowed")
         super(ConvertToInt, self).__init__(option_strings, dest, **kwargs)
         
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(float(values)))

parser = argparse.ArgumentParser(description='Test The BPR with on the fly negative samples')
parser.add_argument('-u','--users',action=ConvertToInt)
parser.add_argument('-i','--items',action=ConvertToInt)
parser.add_argument('--examples', action=ConvertToInt)
parser.add_argument('--batch_size', dest = 'batch_size', type = int,nargs=1,required=True)
parser.add_argument('--num_negs', dest = 'num_negs', type = int,nargs=1,required=True)
parser.add_argument('--num_latent', dest = 'num_latent', type = int,nargs=1,required=True)
parser.add_argument('--source_file', dest = 'source_file', type = str,nargs=1,required=True)
args = parser.parse_args()


def _parse_function(example_proto):
        features = {
            "user":
                            tf.FixedLenFeature((), tf.int64),
            "item": tf.FixedLenFeature((),tf.int64),
            "minItem": tf.FixedLenFeature((),tf.int64),
            "maxItem": tf.FixedLenFeature((),tf.int64)
            }
        parsed_features = tf.parse_example(example_proto, features)
        return parsed_features

    
def input_fn(file_names,batch_size):
    files = tf.data.Dataset.list_files(file_names)
    dataset = files.interleave(tf.data.TFRecordDataset,5,1)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size*5)
    dataset = dataset.map(_parse_function,num_parallel_calls=15)
    #dataset = dataset.cache()
    #dataset = dataset.repeat(5)
    #dataset = dataset.apply(tf.contrib.data.unbatch())
    #dataset = dataset.batch(batch_size)
    return dataset



myDataset = input_fn(args.source_file[0],args.batch_size[0])

iterator = myDataset.make_one_shot_iterator()
next_element = iterator.get_next()

myNeg = Snu(args.num_negs[0])
myNegLayer = myNeg.getSamplingLayer(next_element['item'],next_element['minItem'],
                                                 next_element['maxItem'])

myRanker = Sfr(args.items+1,args.users+1,args.num_latent[0])
itemScore,negScore=myRanker.getFactorProducts(next_element['user'],next_element['item'],myNegLayer)
loss = myRanker.getBPRLoss(itemScore,negScore)


global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(loss,
                                             global_step=global_step)


#config = tf.ConfigProto()
#  config.intra_op_parallelism_threads = 44
#config.inter_op_parallelism_threads = 10
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = time.time()
    for ii in range(args.examples):
        try:
            sess.run([optimizer])
        except tf.errors.OutOfRangeError:
            break
    print('End benchmark', time.time()-t0)
