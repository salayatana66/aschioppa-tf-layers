import tensorflow as tf
import numpy as np
import sys
import time
sXfmax_mod = tf.load_op_library("/home/aschioppa/persistent-disk/aschioppa_tf_layers/cc/shardedXEntSfmax.so")
sXfmax = sXfmax_mod.sharded_xent_sfmax

class Loss:
    def __init__(self,W,b):
        self.W = W
        self.b = b

    def loss_at(self, x, il, iu, ll):
        out = np.zeros_like(iu,dtype=np.float32)
        for i in range(x.shape[0]):
            logits = x[i,:].reshape((1,-1)).dot(self.W[:,il[i] : (iu[i]+1)].reshape((x.shape[1],-1))) + \
             self.b[il[i] : (iu[i]+1)].reshape((1,-1))
            out[i] = Loss.__loss(ll[i]-il[i],logits)
        return out

    # ib = index in the batch
    def pert_loss_W(self, x, il, iu, ll, ib, idWTup, pert):
        tempW = self.W.copy()
        tempW[idWTup[0],idWTup[1]] += pert
        logits = x[ib,:].reshape((1,-1)).dot(tempW[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             self.b[il[ib] : (iu[ib]+1)].reshape((1,-1))
        return Loss.__loss(ll[ib]-il[ib],logits)

    def grad_loss_W(self, x, il, iu, ll):
        out = np.zeros((x.shape[0],x.shape[1],self.W.shape[1]))
        for ib in range(x.shape[0]):
            for a0 in range(x.shape[1]):
                for c0 in range(il[ib], (iu[ib]+1)):
                    loss_plus = self.pert_loss_W(x, il, iu, ll, ib, (a0,c0), 1e-3)
                    loss_minus = self.pert_loss_W(x, il, iu, ll, ib, (a0,c0), -1e-3)
                    out[ib,a0,c0] = (loss_plus-loss_minus)/(2e-3)
        return out

    # ib = index in the batch
    def pert_loss_b(self, x, il, iu, ll, ib, idb, pert):
        tempB = self.b.copy()
        tempB[idb] += pert
        logits = x[ib,:].reshape((1,-1)).dot(self.W[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             tempB[il[ib] : (iu[ib]+1)].reshape((1,-1))
        return Loss.__loss(ll[ib]-il[ib],logits)

    def grad_loss_b(self, x, il, iu, ll):
        out = np.zeros((x.shape[0],self.b.shape[0]))
        for ib in range(x.shape[0]):
            for c0 in range(il[ib], (iu[ib]+1)):
                loss_plus = self.pert_loss_b(x, il, iu, ll, ib, c0, 1e-3)
                loss_minus = self.pert_loss_b(x, il, iu, ll, ib, c0, -1e-3)
                out[ib,c0] = (loss_plus-loss_minus)/(2e-3)
        return out

    # ib = index in the batch
    def pert_loss_x(self, x, il, iu, ll, ib, idx, pert):
        tempX = x.copy()
        tempX[ib,idx] += pert
        logits = tempX[ib,:].reshape((1,-1)).dot(self.W[:,il[ib] : (iu[ib]+1)].reshape((x.shape[1],-1))) + \
             self.b[il[ib] : (iu[ib]+1)].reshape((1,-1))
        return Loss.__loss(ll[ib]-il[ib],logits)

    def grad_loss_x(self, x, il, iu, ll):
        out = np.zeros((x.shape[0],x.shape[1]))
        for ib in range(x.shape[0]):
            for a0 in range(x.shape[1]):
                loss_plus = self.pert_loss_x(x, il, iu, ll, ib, a0, 1e-3)
                loss_minus = self.pert_loss_x(x, il, iu, ll, ib, a0, -1e-3)
                out[ib,a0] = (loss_plus-loss_minus)/(2e-3)
        return out

    def __loss(label,logits):
        _max = np.max(logits)
        _lg = (logits-_max).reshape((-1,))
        _exp = np.exp(_lg)
        _sexp = np.sum(_exp)
        return -np.log(_exp[label]/_sexp)
    

def numpyDense(shape, indices, values):
    out = np.zeros(shape)
    for ii in range(indices.shape[0]):
        #print(indices[ii,:])
        #print(values[ii])
        out[tuple(indices[ii,:])] = values[ii]
    return out

# def sfmax(logits):
#     _max = np.max(logits)
#     _lg = logits-_max
#     _exp = np.exp(_lg)
#     _sexp = np.sum(_exp)
#     return _exp/_sexp

if __name__ == "__main__":
#/--------------------------------
#Valid Tests 
    print("="*32)
    print("Unit Test1 : compare with numpy loss")
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

    myLoss = Loss(W,b)
    print("Numpy: ", myLoss.loss_at(I,lower,upper,labels))
    
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)
        print("C++: ", myOut[0])

    print("="*32)
    print("Unit Test2 : Check correctness of gradient wrt to Weights")
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
    myLoss = Loss(W,b)
    myGrad = myLoss.grad_loss_W(I, lower, upper, labels)
    print("Numpy: ", myGrad)
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)
        print("C++: ", numpyDense(myGrad.shape,myOut[2],myOut[3]))

    print("="*32)
    print("Unit Test3 : Check correctness of gradient wrt to biases")
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
    myLoss = Loss(W,b)
    myGrad = myLoss.grad_loss_b(I, lower, upper, labels)
    print("Numpy: ", myGrad)
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)
        print("C++: ", numpyDense(myGrad.shape,myOut[4],myOut[5]))

    print("="*32)
    print("Unit Test4 : Check correctness of gradient wrt to inputs")
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
    myLoss = Loss(W,b)
    myGrad = myLoss.grad_loss_x(I, lower, upper, labels)
    print("Numpy: ", myGrad)
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)
        print("C++: ", myOut[1])

    print("="*32)
    print("Unit Test5 : test gradients on a large loss problem")
    print("="*32)
    
    np.random.seed(19)
    W = np.random.uniform(low=-1.0,high=1.0,size=7*35).reshape((7,35))
    b = 0.5*np.random.uniform(low=0.0,high=1.0,size=35).reshape((35,))
    lower = np.array(([0]*10)+([20]*10))
    upper = np.array(([19]*10)+([34]*10))
    I = np.random.randn(7*20).reshape((20,7))
    labels1 = np.random.choice([l for l in range(20)],size=10)
    labels2 = np.random.choice([l for l in range(20,35)],size=10)
    labels=np.concatenate([labels1,labels2])

    myLoss = Loss(W,b)
    tW = tf.constant(W,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)

    myGrads = {}
    myGrads['0'] = myLoss.loss_at(I,lower,upper,labels)
    myGrads['W'] = myLoss.grad_loss_W(I,lower,upper,labels)
    myGrads['b'] = myLoss.grad_loss_b(I,lower,upper,labels)
    myGrads['x'] = myLoss.grad_loss_x(I,lower,upper,labels)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        myOut = sess.run(out)

        print('Discrepancy on loss: ' , np.amax(np.array([myGrads['0']-myOut[0]])))
        print('Discrepancy on gradX: ', np.amax(myGrads['x']-myOut[1]))
        print('Discrepancy on gradW: ', np.amax(myGrads['W']-numpyDense(myGrads['W'].shape,myOut[2],myOut[3])))
        print('Discrepancy on gradB: ', np.amax(myGrads['b']-numpyDense(myGrads['b'].shape,myOut[4],myOut[5])))




         
