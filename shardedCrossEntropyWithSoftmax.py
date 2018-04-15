from tensorflow.python.ops import array_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
import tensorflow as tf

"""
Represents a softmax where the set of valid labels
can change across the examples. Valid labels are assumed
to be continuous, and one must supply the inclusive
lower & upper bounds
"""
class ShardedCrossEntropyWithSoftmax:

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

    """
    inputs => what it feeds in the connected layer
    weights => obtained via getWeights
    lowerBound => first value of the valid labels
    upperBound => last value of the valid labels
    labels => the true labels
    parallel_iterations => for the while_loop
    """
    def getLayer(self, inputs, weights, lowerBound, upperBound, labels,
                 parallel_iterations=10):
        # we pull the weights
        W = weights['W']
        b = weights['b']

        # this is the function
        # called in the loop
        # for each item of the batch computes the softmax
        def _fn(x,y):
            lb = lowerBound[x]
            ub = upperBound[x]
            ix = array_ops.reshape(inputs[x],[1,-1])
            lx = labels[x]
            _W = array_ops.slice(W,[0,lb],[-1,ub-lb+1])
            _b = array_ops.reshape(
                array_ops.slice(b,[lb,0],[ub-lb+1,-1]),[-1])                                                                           
            connected = array_ops.reshape(math_ops.matmul(ix,_W)+_b,[-1])
            _max = math_ops.reduce_max(connected)
            _exp = math_ops.exp(connected-_max)
            _sexp = math_ops.reduce_sum(_exp)
            _lexp = array_ops.slice(_exp,[lx-lb],[1])
            _out = math_ops.div(_lexp,_sexp)
            return [tf.add(x,1),y.write(x,_out)]

        # counters for the loop
        n = array_ops.shape(lowerBound)[0]
        i = tf.constant(0)
        # the tensor array is used to produce the output
        # at each iteration in the loop we overwrite it
        accs_ta = tensor_array_ops.TensorArray(dtype=tf.float32, size=n,                                      dynamic_size=False)
                                 
        # loop
        _,output = tf.while_loop(lambda i,_: tf.less(i,n), _fn, [i,accs_ta],
                                 parallel_iterations=parallel_iterations)

        # note before applying operations to output
        # we need to call .stack() to get a tensor
        return -tf.reduce_mean(math_ops.log(output.stack()))

        
        


        
