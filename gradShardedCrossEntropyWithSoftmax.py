import tensorflow as tf
from tensorflow.python.framework import ops

sXfmax_mod = tf.load_op_library("/home/aschioppa/persistent-disk/aschioppa_tf_layers/cc/shardedXEntSfmax.so")
sXfmax = sXfmax_mod.sharded_xent_sfmax
sXGhelper = sXfmax_mod.sharded_xent_sfmax_helper_grad

@ops.RegisterGradient("ShardedXentSfmax")
def _sharded_xent_sfmax_grad(op, *grads):
    loss_grad = grads[0]
    myGrads = sXGhelper(loss_grad,op.outputs[1],op.outputs[2],op.outputs[3],
                        op.outputs[4],op.outputs[5])
    # shapes
    Wshape = tf.shape(op.inputs[1])
    bshape = tf.shape(op.inputs[2])
    return [myGrads[0], tf.IndexedSlices(myGrads[2],
                                         myGrads[1],Wshape),
            tf.IndexedSlices(myGrads[4],
                                         myGrads[3],bshape),
            None,None,None]
