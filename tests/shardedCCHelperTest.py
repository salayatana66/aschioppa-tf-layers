# We test the helper function that computes the gradients for the sharded layer.
import tensorflow as tf
import numpy as np
from shardedCCSoftmaxTest import Loss, numpyDense

sXfmax_mod = tf.load_op_library("/home/aschioppa/persistent-disk/aschioppa_tf_layers/cc/shardedXEntSfmax.so")
sXfmax = sXfmax_mod.sharded_xent_sfmax
sXGhelper = sXfmax_mod.sharded_xent_sfmax_helper_grad


class WeightedLoss:
    def __init__(self,W,b):
        self.Loss = Loss(W,b)

    def loss_at(self, xwg, il, iu, ll):
        return xwg[0].dot(self.Loss.loss_at(xwg[1],il,iu,ll))

        # ib = index in the batch
    def pert_loss_W(self, xwg, il, iu, ll, ib, idWTup, pert):
        x = xwg[1].copy()
        tempW = self.Loss.W.copy()
        tempW[idWTup[0],idWTup[1]] += pert
        logits = x[ib,:].reshape((1,-1)).dot(tempW[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             self.Loss.b[il[ib] : (iu[ib]+1)].reshape((1,-1))
        # as Loss.__loss is now private we need to explicitly
        # prepend the class name hence Loss._Loss__loss
        return xwg[0][ib]*Loss._Loss__loss(ll[ib]-il[ib],logits)

    def grad_loss_W(self, xwg, il, iu, ll):
        x = xwg[1].copy()
        out = np.zeros((x.shape[0],x.shape[1],self.Loss.W.shape[1]))
        for ib in range(x.shape[0]):
            for a0 in range(x.shape[1]):
                for c0 in range(il[ib], (iu[ib]+1)):
                    loss_plus = self.pert_loss_W(xwg, il, iu, ll, ib, (a0,c0), 1e-3)
                    loss_minus = self.pert_loss_W(xwg, il, iu, ll, ib, (a0,c0), -1e-3)
                    out[ib,a0,c0] = (loss_plus-loss_minus)/(2e-3)
        return np.sum(out,0)

        # ib = index in the batch
    def pert_loss_b(self, xwg, il, iu, ll, ib, idb, pert):
        x=xwg[1].copy()
        tempB = self.Loss.b.copy()
        tempB[idb] += pert
        logits = xwg[1][ib,:].reshape((1,-1)).dot(self.Loss.W[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             tempB[il[ib] : (iu[ib]+1)].reshape((1,-1))
        return xwg[0][ib]*Loss._Loss__loss(ll[ib]-il[ib],logits)

    def grad_loss_b(self, xwg, il, iu, ll):
        x = xwg[1].copy()
        out = np.zeros((x.shape[0],self.Loss.b.shape[0]))
        for ib in range(x.shape[0]):
            for c0 in range(il[ib], (iu[ib]+1)):
                loss_plus = self.pert_loss_b(xwg, il, iu, ll, ib, c0, 1e-3)
                loss_minus = self.pert_loss_b(xwg, il, iu, ll, ib, c0, -1e-3)
                out[ib,c0] = (loss_plus-loss_minus)/(2e-3)
        return np.sum(out,0)

        # ib = index in the batch
    def pert_loss_x(self, xwg, il, iu, ll, ib, idx, pert):
        x=xwg[1].copy()
        tempX = xwg[1].copy()
        tempX[ib,idx] += pert
        logits = tempX[ib,:].reshape((1,-1)).dot(self.Loss.W[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             self.Loss.b[il[ib] : (iu[ib]+1)].reshape((1,-1))
        return xwg[0][ib]*Loss._Loss__loss(ll[ib]-il[ib],logits)

    def grad_loss_x(self, xwg, il, iu, ll):
        x=xwg[1].copy()
        out = np.zeros((x.shape[0],x.shape[1]))
        for ib in range(x.shape[0]):
            for a0 in range(x.shape[1]):
                loss_plus = self.pert_loss_x(xwg, il, iu, ll, ib, a0, 1e-3)
                loss_minus = self.pert_loss_x(xwg, il, iu, ll, ib, a0, -1e-3)
                out[ib,a0] = (loss_plus-loss_minus)/(2e-3)
        return out

def indexSlicesDense(shape, indices, values):
    out = np.zeros(shape)
    for ii in range(indices.shape[0]):
        if len(values.shape) > 1:
            out[indices[ii],:] = values[ii,:]
        else:
            out[indices[ii]] = values[ii]
    return out

if __name__ == "__main__":
    print("="*32)
    print("Unit Test : Compare with weighted numpy loss")
    print("="*32)

    W = np.array([[0.4,0.32,-1.0,0.5,1.17],
                   [0.3,0.25,-1.5,0.4,1.25],
                   [0.2,0.19,-1.3,0.33,1.33]])
    b = np.array([-1.20,-1.3,1.4,
                   -17.3,-19.2])
    lower = np.array([0,3])
    upper = np.array([2,4])
    labels = np.array([2,4])
    I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])
    weights = np.array([2.3,4.5])
    tW = tf.constant(W.T,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    tGl = tf.constant(weights,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    myLoss = WeightedLoss(W,b)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)
        myGrads = sess.run(sXGhelper(tGl,out[1],out[2],out[3],out[4],out[5]))
        print("C++: ", np.tensordot(weights,myOut[0],1))
        print("Numpy: ", myLoss.loss_at((weights,I),lower,upper,labels))
        print("Numpy: ", myLoss.grad_loss_W((weights,I),lower,upper,labels))
        print("C++: ", indexSlicesDense(W.T.shape,myGrads[1],myGrads[2]))
        print("C++: ", 
        indexSlicesDense(b.shape,myGrads[3],myGrads[4]))
        print("Numpy: ", myLoss.grad_loss_b((weights,I),lower,upper,labels))
        print("C++: ", myGrads[0])
        print("Numpy: ", myLoss.grad_loss_x((weights,I),lower,upper,labels))
