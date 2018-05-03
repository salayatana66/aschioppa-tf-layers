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
        tempW = W.copy()
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
        tempB = b.copy()
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
    print("="*32)
    print("Unit Test5 : test gradients on a large loss problem")
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
        # print(myGrads['b'])
        # print(myGrads['b'].shape)
        # print(numpyDense(myGrads['b'].shape,myOut[4],myOut[5]).shape)
        # print(numpyDense(myGrads['b'].shape,myOut[4],myOut[5]))
        # print(myOut[4])
        #print(numpyDense(myGrads['b'].shape,myOut[4],myOut[5]))
    # # _data = []
    # # _verify_data = []
    # # input_placeholder = tf.placeholder(tf.float32,shape=[None,3])
    # # label_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # lower_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # upper_placeholder = tf.placeholder(tf.int32,shape=[None])

    # # for _ in range(10000):
    # #     _bx = []
    # #     _y1 = []
    # #     _y2 = []
    # #     _p1 = []
    # #     _p2 = []
    # #     for j in range(25):
    # #         x=np.reshape(np.random.ranf(3),[1,-1])
    # #         _bx.append(x)
    # #         lg1=np.dot(x,W[:,0:3])+np.reshape(b[0:3,:],[-1])
    # #         lg2=np.dot(x,W[:,3:])+np.reshape(b[3:,:],[-1])
    # #         p1 = sfmax(lg1)
    # #         _p1.append(p1)
    # #         p2 = sfmax(lg2)
    # #         _p2.append(p2)
    # #         y1 = np.where(np.random.multinomial(1,p1.flatten())>0)[0][0]
    # #         _y1.append(y1)
    # #         y2 = np.where(np.random.multinomial(1,p2.flatten())>0)[0][0]
    # #         _y2.append(y2)
    # #     _data.append({input_placeholder: np.concatenate(_bx+_bx,axis=0),
    # #                   label_placeholder : np.concatenate([np.array(_y1),
    # #                                             3+np.array(_y2)],
    # #                                            axis=0),
    # #                   lower_placeholder : np.concatenate([np.zeros((len(_y1),),
    # #                                                      dtype=np.int32),
    # #                           3*np.ones((len(_y2),),
    # #                                          dtype=np.int32)],axis=0),
    # #                   upper_placeholder : np.concatenate([2*np.ones((len(_y1),),
    # #                                                       dtype=np.int32),
    # #                           4*np.ones((len(_y2),),
    # #                                          dtype=np.int32)],axis=0)})
    # #     _verify_data.append({
    # #                  'p1' : np.concatenate(_p1,axis=0),
    # #                  'p2': np.concatenate(_p2,axis=0),
    # #                  'x' : np.concatenate(_bx,axis=0)})

#/--------------------------------
# Valid Tests 
    # print("="*32)
    # print("Unit Test1 : compare with numpy loss")
    # print("="*32)

    # W = np.array([[0.4,0.32,-1.0,0.5,1.17],
    #                [0.3,0.25,-1.5,0.4,1.25],
    #                [0.2,0.19,-1.3,0.33,1.33]])
    # b = np.array([-1.20,-1.3,1.4,
    #                -17.3,-19.2])
    # lower = np.array([0,3])
    # upper = np.array([2,4])
    # labels = np.array([2,4])
    # I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])

    # myLoss = Loss(W,b)
    # print("Numpy: ", myLoss.loss_at(I,lower,upper,labels))
    
    # tW = tf.constant(W,dtype=tf.float32)
    # tb = tf.constant(b,dtype=tf.float32)
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     myOut = sess.run(out)
    #     print("C++: ", myOut[0])

    # print("="*32)
    # print("Unit Test2 : Check correctness of gradient wrt to Weights")
    # print("="*32)

    # W = np.array([[0.4,0.32,-1.0,0.5,1.17],
    #                [0.3,0.25,-1.5,0.4,1.25],
    #                [0.2,0.19,-1.3,0.33,1.33]])
    # b = np.array([-1.20,-1.3,1.4,
    #                -17.3,-19.2])
    # lower = np.array([0,3])
    # upper = np.array([2,4])
    # labels = np.array([2,4])
    # I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])
    # myLoss = Loss(W,b)
    # myGrad = myLoss.grad_loss_W(I, lower, upper, labels)
    # print("Numpy: ", myGrad)
    # tW = tf.constant(W,dtype=tf.float32)
    # tb = tf.constant(b,dtype=tf.float32)
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     myOut = sess.run(out)
    #     print("C++: ", numpyDense(myGrad.shape,myOut[2],myOut[3]))

    # print("="*32)
    # print("Unit Test3 : Check correctness of gradient wrt to biases")
    # print("="*32)

    # W = np.array([[0.4,0.32,-1.0,0.5,1.17],
    #                [0.3,0.25,-1.5,0.4,1.25],
    #                [0.2,0.19,-1.3,0.33,1.33]])
    # b = np.array([-1.20,-1.3,1.4,
    #                -17.3,-19.2])
    # lower = np.array([0,3])
    # upper = np.array([2,4])
    # labels = np.array([2,4])
    # I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])
    # myLoss = Loss(W,b)
    # myGrad = myLoss.grad_loss_b(I, lower, upper, labels)
    # print("Numpy: ", myGrad)
    # tW = tf.constant(W,dtype=tf.float32)
    # tb = tf.constant(b,dtype=tf.float32)
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     myOut = sess.run(out)
    #     print("C++: ", numpyDense(myGrad.shape,myOut[4],myOut[5]))

    # print("="*32)
    # print("Unit Test4 : Check correctness of gradient wrt to inputs")
    # print("="*32)

    # W = np.array([[0.4,0.32,-1.0,0.5,1.17],
    #                [0.3,0.25,-1.5,0.4,1.25],
    #                [0.2,0.19,-1.3,0.33,1.33]])
    # b = np.array([-1.20,-1.3,1.4,
    #                -17.3,-19.2])
    # lower = np.array([0,3])
    # upper = np.array([2,4])
    # labels = np.array([2,4])
    # I = np.array([[1.0,1.0,1.0],[0.4,0.4,-1.0]])
    # myLoss = Loss(W,b)
    # myGrad = myLoss.grad_loss_x(I, lower, upper, labels)
    # print("Numpy: ", myGrad)
    # tW = tf.constant(W,dtype=tf.float32)
    # tb = tf.constant(b,dtype=tf.float32)
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     myOut = sess.run(out)
    #     print("C++: ", myOut[1])

########################################################################    
    # print("="*32)
    # print("Unit Test1 : compare with numpy loss")
    # print("="*32)
    # W = np.array([[0.4,0.32,-1.0,0.5,1.17],
    #                [0.3,0.25,-1.5,0.4,1.25],
    #                [0.2,0.19,-1.3,0.33,1.33]])
    # tW = tf.constant(W,dtype=tf.float32)
    # b = np.array([-1.20,-1.3,1.4,
    #                -17.3,-19.2])
    # tb = tf.constant(b,dtype=tf.float32)
    # lower = np.array([0,3])
    # upper = np.array([2,4])
    # labels = np.array([2,4])
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # I = np.array([[-1.0,-1.0,-1.0],[1.0,1.0,1.0]])
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # logit1 = np.dot(I[0,:],W[:,0:3])+np.reshape(b[0:3],[-1])
    # logit2 = np.dot(I[1,:],W[:,3:])+np.reshape(b[3:],[-1])
    # loss1 = loss(labels[1]-3,logit2)
    # loss0 = loss(labels[0], logit1)
    # print('Python1 ', loss0)
    # print('Python2 ', loss1)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print("TF(C++) : ", sess.run(out))

    # print("="*32)
    # print("Unit Test2 : bigger loss on W")
    # print("="*32)
    
    # np.random.seed(0)
    # W = np.random.uniform(low=-1.0,high=1.0,size=7*35).reshape((7,35))
    # b = 0.5*np.random.uniform(low=0.0,high=1.0,size=35).reshape((35,))
    # lower = np.array(([0]*10)+([20]*10))
    # upper = np.array(([19]*10)+([34]*10))
    # I = np.random.randn(7*20).reshape((20,7))
    # labels1 = np.random.choice([l for l in range(20)],size=10)
    # labels2 = np.random.choice([l for l in range(20,35)],size=10)
    # labels=np.concatenate([labels1,labels2])
    # logits1 = I[:10,:].dot(W[:,:20])+b[:20]
    # logits2 = I[10:,:].dot(W[:,20:])+b[20:]
    # tW = tf.constant(W,dtype=tf.float32)
    # tb = tf.constant(b,dtype=tf.float32)
    # tlower = tf.constant(lower,dtype=tf.int32)
    # tupper = tf.constant(upper,dtype=tf.int32)
    # tlabels = tf.constant(labels,dtype=tf.int32)
    # inputs = tf.constant(I,dtype=tf.float32)
    # out = sXfmax(inputs,tW,tb,tlower,tupper,tlabels)
    # pyLoss0 = [loss(labels[i],logits1[i,:]) for i in range(10)]
    # pyLoss1 = [loss(labels[i]-20,logits2[i-10,:]) for i in range(10,20)]
    # print("Python")
    # print(np.concatenate([pyLoss0,pyLoss1]))
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     print("TF(C++) : ", sess.run(out))

    
    # # _data = []
    # # _verify_data = []
    # # input_placeholder = tf.placeholder(tf.float32,shape=[None,3])
    # # label_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # lower_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # upper_placeholder = tf.placeholder(tf.int32,shape=[None])

    # # for _ in range(10000):
    # #     _bx = []
    # #     _y1 = []
    # #     _y2 = []
    # #     _p1 = []
    # #     _p2 = []
    # #     for j in range(25):
    # #         x=np.reshape(np.random.ranf(3),[1,-1])
    # #         _bx.append(x)
    # #         lg1=np.dot(x,W[:,0:3])+np.reshape(b[0:3,:],[-1])
    # #         lg2=np.dot(x,W[:,3:])+np.reshape(b[3:,:],[-1])
    # #         p1 = sfmax(lg1)
    # #         _p1.append(p1)
    # #         p2 = sfmax(lg2)
    # #         _p2.append(p2)
    # #         y1 = np.where(np.random.multinomial(1,p1.flatten())>0)[0][0]
    # #         _y1.append(y1)
    # #         y2 = np.where(np.random.multinomial(1,p2.flatten())>0)[0][0]
    # #         _y2.append(y2)
    # #     _data.append({input_placeholder: np.concatenate(_bx+_bx,axis=0),
    # #                   label_placeholder : np.concatenate([np.array(_y1),
    # #                                             3+np.array(_y2)],
    # #                                            axis=0),
    # #                   lower_placeholder : np.concatenate([np.zeros((len(_y1),),
    # #                                                      dtype=np.int32),
    # #                           3*np.ones((len(_y2),),
    # #                                          dtype=np.int32)],axis=0),
    # #                   upper_placeholder : np.concatenate([2*np.ones((len(_y1),),
    # #                                                       dtype=np.int32),
    # #                           4*np.ones((len(_y2),),
    # #                                          dtype=np.int32)],axis=0)})
    # #     _verify_data.append({
    # #                  'p1' : np.concatenate(_p1,axis=0),
    # #                  'p2': np.concatenate(_p2,axis=0),
    # #                  'x' : np.concatenate(_bx,axis=0)})

    # # with tf.variable_scope("UnitTest2"):
    # #     trainableWeights = sf.getWeights()
    # # myLoss = sf.getLayer(input_placeholder,trainableWeights,
    # #                      lower_placeholder,
    # #                      upper_placeholder,label_placeholder)
    # # opt = tf.train.GradientDescentOptimizer(1)
    # # train_op = opt.minimize(myLoss)
    # # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    # #     for j in range(2):
    # #         losses = []
    # #         for i,_batch in enumerate(_data):
    # #             loss, _ = sess.run([myLoss,train_op],feed_dict=_batch)
    # #             losses.append(loss)
                
    # #         print(sess.run(trainableWeights))
    # #         print(sess.run(weights))
    # #         print(j, np.mean(losses))
    # #     newW = sess.run(trainableWeights)
    # #     mean1 = []
    # #     mean2 = []
    # #     for j in range(20):
    # #         _b = _verify_data[j]
    # #         logits=_b['x'].dot(newW['W'])+np.reshape(newW['b'],[-1])
    # #         _P1 = []
    # #         _P2 = []
    # #         for j in range(logits.shape[0]):
    # #             _P1.append(np.reshape(sfmax(logits[j,:3]),(1,-1)))
    # #             _P2.append(np.reshape(sfmax(logits[j,3:]),(1,-1)))
    # #         mean1.append(np.mean(np.abs(np.concatenate(_P1,axis=0)-_b['p1'])))
    # #         mean2.append(np.mean(np.abs(np.concatenate(_P2,axis=0)-_b['p2'])))

    # #     print('Delta1' , np.mean(mean1))
    # #     print('Delta2' , np.mean(mean2))

    # # print("="*32)
    # # print("Unit Test3 : Speed of training to the native softmax in C++")
    # # print("="*32)
    # # np.random.seed(0)
    # # sf1 = ShardedCrossEntropyWithSoftmax(3,4)
    # # _data = []
    # # W1 = np.array([[0.4,0.32,-1.0,0.5],
    # #               [0.3,0.25,-1.5,0.4],
    # #               [0.2,0.19,-1.3,0.33]])
    # # b1 = np.array([[-1.20],[-1.3],[1.4],
    # #               [-1.3]])
    # # x_placeholder = tf.placeholder(tf.float32,shape=[None,3])
    # # y1_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # lab_placeholder = tf.placeholder(tf.int32,shape=[None,4])
    # # l_placeholder = tf.placeholder(tf.int32,shape=[None])
    # # u_placeholder = tf.placeholder(tf.int32,shape=[None])

    # # for _ in range(1000):
    # #     _bx = []
    # #     _y1 = []
    # #     _p1 = []
    # #     _lab = []
    # #     for j in range(25):
    # #         x=np.reshape(np.random.ranf(3),[1,-1])
    # #         _bx.append(x)
    # #         lg1=np.dot(x,W1)+np.reshape(b1,[-1])
    # #         p1 = sfmax(lg1)
    # #         lab = np.reshape(np.array(np.random.multinomial(1,p1.flatten())),
    # #                          (1,-1))
    # #         _lab.append(lab)
    # #         y1 = np.where(lab>0)[0][0]
    # #         _y1.append(y1)
    # #     _data.append({x_placeholder: np.concatenate(_bx,axis=0),
    # #                   y1_placeholder : np.array(_y1),
    # #                   l_placeholder : np.zeros((len(_y1),),
    # #                                                      dtype=np.int32),
    # #                   u_placeholder : 3*np.ones((len(_y1),),
    # #                                                       dtype=np.int32),
    # #                   lab_placeholder : np.concatenate(_lab,axis=0)})
    # # print(np.shape(np.concatenate(_lab,axis=0)))
    
    # # with tf.variable_scope("UnitTest3"):
    # #     trainableWeights1 = sf1.getWeights(seed=0)
    # #     tW1 = tf.get_variable('tW1',dtype = tf.float32,initializer=
    # #                           tf.random_uniform(
    # #         shape=(3,4), minval=-1.0,maxval=1.0,seed=0))
    # #     tb1 = tf.get_variable('tb1',initializer=tf.zeros(shape=(4,),
    # #                                                      dtype=tf.float32))
    # #     clayer = tf.reshape(tf.matmul(x_placeholder,tW1),(-1,4))+tb1
    # # myLoss1 = sf.getLayer(x_placeholder,trainableWeights1,
    # #                      l_placeholder,
    # #                       u_placeholder,y1_placeholder,parallel_iterations=25)
    # # opt1 = tf.train.GradientDescentOptimizer(10)
    # # train_op1 = opt1.minimize(myLoss1)
    
    # # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    # #     t0 = time.time()
    # #     for j in range(2):
    # #         losses = []
    # #         for i,_batch in enumerate(_data):
    # #             loss, _ = sess.run([myLoss1,train_op1],feed_dict=_batch)
    # #             losses.append(loss)
    # #         print(j, np.mean(losses))
    # #     t1 = time.time()
    # #     delta1 = t1-t0
    # # tfLoss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1_placeholder,     logits=clayer)
    # # opt2 = tf.train.GradientDescentOptimizer(1e-1)
    # # train_op2 = opt2.minimize(tfLoss1)

    # # with tf.Session() as sess:
    # #     sess.run(tf.global_variables_initializer())
    # #     t2 = time.time()
    # #     for j in range(2):
    # #         losses = []
    # #         for i,_batch in enumerate(_data):
    # #             loss, _ = sess.run([tfLoss1,train_op2],feed_dict=_batch)
    # #             losses.append(loss)
    # #         print(j, np.mean(losses))
    # #     t3 = time.time()
    # #     delta2 = t3-t2

    # # print('Sharded : ', delta1)
    # # print('Native : ', delta2)
    # # print('Ratio : ', delta1/delta2)




         
