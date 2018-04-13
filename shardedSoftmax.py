from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
import tensorflow as tf

# partitioned fully connected layer
class ShardedSoftmax:

    def __init__(self, numInputs, numOutputs):
        self.numInputs = numInputs
        self.numOutputs = numOutputs
        
    def getWeights(self,initializers = {'W' : None, 'b' : None}, seed = None):
        if seed is None:
            seed = 0
        _initializers = {}
        if initializers['W'] is None:
            _initializers['W'] = tf.random_uniform(shape=(self.numInputs,self.numOutputs),
                                                   minval=-1.0,maxval=1.0,dtype=tf.float32,seed=seed)
        else:
            _initializers['W'] = initializers['W']

        if initializers['b'] is None:
            _initializers['b'] = tf.zeros(shape=(self.numOutputs,1))
        else:
            _initializers['b'] = initializers['b']

        outTensors = {}
        outTensors['W'] = tf.get_variable("W",dtype=tf.float32,initializer=_initializers['W'])
        outTensors['b'] = tf.get_variable("b",dtype=tf.float32,initializer=_initializers['b'])

        return outTensors
                
    def getLayer(self, inputs, weights, lowerBound, upperBound):
        # here somewhere define weights
        W = weights['W']
        b = weights['b']

        #elems = (lb, ub)
        def _fn(x,y):
            lb = lowerBound[x]
            ub = upperBound[x]
            ix = array_ops.reshape(inputs[x],[1,-1])
            _W = array_ops.slice(W,[0,lb],[-1,ub-lb+1])
            _b = array_ops.slice(b,[lb,0],[ub-lb+1,-1])
            out = tf.reduce_sum(math_ops.matmul(ix,_W)+_b)
            return [tf.add(x,1),y.write(x,out)]

        
        #n = tf.shape(lowerBound)[0]
        n = array_ops.shape(lowerBound)[0]
        i = tf.constant(0)
        accs_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=n,                                                                                      dynamic_size=False)
                                 
        # just to preserve the loop invariant
        _,output = tf.while_loop(lambda i,_: tf.less(i,n), _fn, [i,accs_ta])#,[i.get_shape()])#,

        #
        #tf.TensorShape([None,None])])
        return output.stack()
        #return tf.map_fn(_fn, tf.range(tf.constant(0),tf.shape(lowerBound)[0]),dtype=tf.float32)
        
        


        
