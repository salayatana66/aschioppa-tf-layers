import tensorflow as tf
import numpy as np
import pandas as pd
from aschioppa_tf_layers.latentLayers import latentUnashedCategorical as Lcat

cat1 = ["a", "b", "c"]
cat2 = [11, 9, 27, -1]

lcat1 = Lcat(cat1, default_index=0, key_type=tf.string, num_latent_factors=4,
             prior_weights = np.array([[0.1, 0.01, 0.4, 0.5],
                                         [-0.2,0.02,-0.4,0.5],
                                         [-0.22,0.5,0.4,0.1]]))

lcat2 = Lcat(cat2, default_index=3, key_type=tf.int64, num_latent_factors=4)

input1 = tf.constant(["a","a","b","b","c","c","e","e"],dtype=tf.string)
input2 = tf.constant([[11, 9], [27, 27], [-1, -1], [13, 17]],dtype=tf.int64)

look1 = lcat1.getLookupLayer(input1)
look2 = lcat2.getLookupLayer(input2)
invlook1 = lcat1.getInverseLookupLayer(look1)
invlook2 = lcat2.getInverseLookupLayer(look2)
latentLayer1 = lcat1.getLatentLayer(input1)
latentLayer1bis = lcat1.getLatentLayer(look1,withLookup=False)
latentLayer2 = lcat2.getLatentLayer(input2)
latentLayer2bis = lcat2.getLatentLayer(look2,withLookup=False)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    print( sess.run([look1]))
    print (sess.run([invlook1]))
    print (sess.run([latentLayer1]))
    print (sess.run([latentLayer1bis]))
    print (sess.run([look2]))
    print (sess.run([invlook2]))
    print (sess.run([latentLayer2]))
    print (sess.run([latentLayer2bis]))
    print (Lcat.numInstances)


