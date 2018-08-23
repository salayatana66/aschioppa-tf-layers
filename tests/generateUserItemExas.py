import tensorflow as tf
import numpy as np
import sys
import argparse

class ConvertToInt(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
         if nargs is not None:
             raise ValueError("nargs not allowed")
         super(ConvertToInt, self).__init__(option_strings, dest, **kwargs)
         
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, int(float(values)))

parser = argparse.ArgumentParser(description='Generate User Item synthetic data')
parser.add_argument('-u','--users',action=ConvertToInt)
parser.add_argument('-i','--items',action=ConvertToInt)
parser.add_argument('-e','--examples', action = ConvertToInt)
parser.add_argument('-f','--dest_file',dest='dest_file', type = str)
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
    np.random.seed(12)
    # sample users
    users = np.random.random_integers(low = 0, high = args.users,
                                      size = args.examples)
    
    # sample items
    items = np.sort(np.reshape(np.random.random_integers(low = 0, high = args.items,
                                                         size = args.examples*3),[-1,3]),axis=1)

    
    writer = tf.python_io.TFRecordWriter(args.dest_file)

    for u, i1, i2, i3 in np.nditer([users,items[:,0],items[:,1],items[:,2]]):
        features = tf.train.Features(feature={'user' : _int64_feature(u),
                                              'item' : _int64_feature(i2),
                                              'minItem' : _int64_feature(i1),
                                              'maxItem' : _int64_feature(i3)})
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)

    writer.close()




