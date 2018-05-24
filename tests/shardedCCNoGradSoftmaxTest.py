# We test the sharded layer without the gradient operations;
# Note that this can be used at prediction time as it's faster without the gradient computation.
import tensorflow as tf
import numpy as np
import sys
import time
sXfmax_mod = tf.load_op_library("/home/aschioppa/persistent-disk/aschioppa_tf_layers/cc/shardedXEntSfmax.so")
sXfmax = sXfmax_mod.sharded_xent_sfmax_no_grad
def loss(label,logits):
    _max = np.max(logits)
    _lg = logits-_max
    _exp = np.exp(_lg)
    _sexp = np.sum(_exp)

    return -np.log(_exp[label]/_sexp)

def sfmax(logits):
    _max = np.max(logits)
    _lg = logits-_max
    _exp = np.exp(_lg)
    _sexp = np.sum(_exp)
    return _exp/_sexp

if __name__ == "__main__":
    print("="*32)
    print("Unit Test1 : compare with numpy loss")
    print("="*32)
    W = np.array([[0.4,0.32,-1.0,0.5,1.17],
                   [0.3,0.25,-1.5,0.4,1.25],
                   [0.2,0.19,-1.3,0.33,1.33]])
    tW = tf.constant(W.T,dtype=tf.float32)
    b = np.array([-1.20,-1.3,1.4,
                   -17.3,-19.2])
    tb = tf.constant(b,dtype=tf.float32)
    lower = np.array([0,3])
    upper = np.array([2,4])
    labels = np.array([2,4])
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    I = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]])
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    logit1 = np.dot(I[0,:],W[:,0:3])+np.reshape(b[0:3],[-1])
    logit2 = np.dot(I[1,:],W[:,3:])+np.reshape(b[3:],[-1])
    loss1 = loss(labels[1]-3,logit2)
    loss0 = loss(labels[0], logit1)
    print('Python1 ', loss0)
    print('Python2 ', loss1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("TF(C++) : ", sess.run(out))

    print("="*32)
    print("Unit Test2 : bigger loss on W")
    print("="*32)
    
    np.random.seed(0)
    W = np.random.uniform(low=-1.0,high=1.0,size=7*35).reshape((7,35))
    b = 0.5*np.random.uniform(low=0.0,high=1.0,size=35).reshape((35,))
    lower = np.array(([0]*10)+([20]*10))
    upper = np.array(([19]*10)+([34]*10))
    I = np.random.randn(7*20).reshape((20,7))
    labels1 = np.random.choice([l for l in range(20)],size=10)
    labels2 = np.random.choice([l for l in range(20,35)],size=10)
    labels=np.concatenate([labels1,labels2])
    logits1 = I[:10,:].dot(W[:,:20])+b[:20]
    logits2 = I[10:,:].dot(W[:,20:])+b[20:]
    tW = tf.constant(W.T,dtype=tf.float32)
    tb = tf.constant(b,dtype=tf.float32)
    tlower = tf.constant(lower,dtype=tf.int32)
    tupper = tf.constant(upper,dtype=tf.int32)
    tlabels = tf.constant(labels,dtype=tf.int32)
    inputs = tf.constant(I,dtype=tf.float32)
    out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    pyLoss0 = [loss(labels[i],logits1[i,:]) for i in range(10)]
    pyLoss1 = [loss(labels[i]-20,logits2[i-10,:]) for i in range(10,20)]
    print("Python")
    print(np.concatenate([pyLoss0,pyLoss1]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("TF(C++) : ", sess.run(out))
