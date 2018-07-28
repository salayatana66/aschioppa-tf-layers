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

if __name__ == "__main__":
    #t0 = time.time()
    np.random.seed(12)
    # sample users
    users = np.random.random_integers(low = 0, high = args.users[0],
                                      size = args.examples[0])
    #print(users)
    # sample items
    items = np.reshape(np.random.random_integers(low = 0, high = args.items[0],
                                          size = args.examples[0]*3),[-1,3])

    #print(items)
    #print((time.time()-t0)/60)
    writer = tf.python_io.TFRecordWriter(args.dest_file[0])
    for u, i1, i2, i3 in np.nditer([users,items[:,0],items[:,1],items[:,2]]):
        ilist = sorted([i1,i2,i3])
        features = tf.train.Features(feature={'user' : ???
                                              'item' : ilist[1],
                                              'minItem' : ilist[0],
                                              'maxItem' : ilist[2]})
        example = tf.train.Example(features=features)
        serialized = example.SerializeToString()
        writer.write(serialized)
