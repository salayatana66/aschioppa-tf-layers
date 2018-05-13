import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
import numpy as np

sXfmax_mod = tf.load_op_library("/home/aschioppa/persistent-disk/aschioppa_tf_layers/cc/shardedXEntSfmax.so")
sXfmax = sXfmax_mod.sharded_xent_sfmax
sXGhelper = sXfmax_mod.sharded_xent_sfmax_helper_grad

@ops.RegisterGradient("ShardedXentSfmax")
def _sharded_xent_sfmax_grad(op, grad):
    loss_grad = grad[0]
    # do we need the reshape?
    gradI = math_ops.matmul(array_ops.reshape(loss_grad,[1,-1]),
                            op.outputs[1])
    gradW = None
    gradB = None
    return [gradI, gradW, gradB, None, None, None]


def numpyDense(shape, indices, values):
    out = np.zeros(shape)
    for ii in range(indices.shape[0]):
        out[tuple(indices[ii,:])] = values[ii]
    return out


if __name__ == "__main__":
    W = np.array([[0.4,0.32,-1.0,0.5,1.17],
                   [0.3,0.25,-1.5,0.4,1.25],
                   [0.2,0.19,-1.3,0.33,1.33]])
    b = np.array([-1.20,-1.3,1.4,
                   -17.3,-19.2])
    lower = np.array([0,3])
    upper = np.array([2,4])
    labels = np.array([2,4])
    I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])
    gL = np.array([1.0,1.0])
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    tGl = tf.constant(gL,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #dI = sess.run(math_ops.matmul(array_ops.reshape(tGl,[1,-1]),out[1]))
        myOut = sess.run(out)
        print("Python: ", gL.reshape([-1,1])*myOut[1])
        myGrads = sess.run(sXGhelper(tGl,out[1],out[2],out[3],out[4],out[5]))
        print("C++: ", myGrads[0])
        print("Python: ", np.tensordot(gL,
                                       numpyDense((2,W.shape[0],W.shape[1]),
                                                  myOut[2],myOut[3]),1))
        print("C++: ", 
        numpyDense(W.shape,myGrads[1],myGrads[2]))
        print("Python: ", np.tensordot(gL,
                                       numpyDense((2,b.shape[0]),
                                                  myOut[4],myOut[5]),1))
        print("C++: ", 
        numpyDense(b.shape,myGrads[3],myGrads[4]))

