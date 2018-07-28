import tensorflow as tf
import numpy as np
import sys
print(sys.path)
import time
import argparse

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('--users', dest = 'users',type=int, nargs=1,required=True)
parser.add_argument('--items', dest = 'items',type=int,nargs=1,required=True)
parser.add_argument('--examples', dest = 'examples', type = int, nargs=1,required=True)
parser.add_argument('--dest_file', dest = 'dest_file', type = str,nargs=1,required=True)
args = parser.parse_args()

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _parse_function(example_proto):
        features = {
            "user":
                            tf.FixedLenFeature((), tf.int64),
            "item": tf.FixedLenFeature((),tf.int64),
            "minItem": tf.FixedLenFeature((),tf.int64),
            "maxItem": tf.FixedLenFeature((),tf.int64)
            }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features

if __name__ == "__main__":
    t0 = time.time()

    np.random.seed(12)
    # sample users
    users = np.random.random_integers(low = 0, high = args.users[0],
                                      size = args.examples[0])
    
    # sample items
    items = np.reshape(np.random.random_integers(low = 0, high = args.items[0],
                                          size = args.examples[0]*3),[-1,3])

    
    writer = tf.python_io.TFRecordWriter(args.dest_file[0])

    for u, i1, i2, i3 in np.nditer([users,items[:,0],items[:,1],items[:,2]]):
        ilist = sorted([i1,i2,i3])
        features = tf.train.Features(feature={'user' : _int64_feature(u),
                                              'item' : _int64_feature(ilist[1]),
                                              'minItem' : _int64_feature(ilist[0]),
                                              'maxItem' : _int64_feature(ilist[2])})
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    writer.close()
    print("End of writing ", args.examples[0], " rows")
    print((time.time()-t0)/60)
    # for testing reading back
    # dataset = (tf.data.TFRecordDataset(args.dest_file[0])
    #            .map(_parse_function)
    #            .batch(1))
    # iterator = dataset.make_one_shot_iterator()
    # next_element = iterator.get_next()

    # with tf.Session() as sess:
    #     for i in range(4):
    #         print(sess.run([next_element]))


