import tensorflow as tf
import input_data
import sys
import numpy as np
import time
start = time.time()

numberlabels = 2
hiddenunits1 = 100
lamb = 0.01  # regularization parameter
batchsize_test = 10000
learning_rate = 0.0001
batch_size = 1024
trainstep = 20000

def weight_variable(shape):
    initial = tf.truncated_normal(
        shape, stddev=0.01) 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(
        0.01, shape=shape)  #shape=[2]     [[0.01 0.01] [0.01 0.01]]
    return tf.Variable(initial)

# defining the layers
def layers(x, W, b):
    return tf.nn.sigmoid(tf.matmul(x, W)+b)

sess = tf.Session()

LENGTH_GET = [80]
STEP = [[1,81]] 
# LENGTH_GET = [8,16,32,48,64]
# STEP = [[1,9],[1,17],[1,33],[1,49],[1,65]] 


# STEP = [[1, TIME + 1], [1, TIME + 1]]  # or Step = [10]

  # or Step = [10]

for size in range(len(LENGTH_GET)):
   
    # Size
    lx = LENGTH_GET[size]

    # Time
    ly = STEP[size][1] - STEP[size][0]

    # defining the model

    #first layer
    #weights and bias
    W_1 = weight_variable([lx * ly, hiddenunits1])
    b_1 = bias_variable([hiddenunits1])

    #Apply a sigmoid
    #x is input_data, y_ is the label
    x = tf.placeholder("float", shape=[None, lx * ly])   
    y_ = tf.placeholder("float", shape=[None, numberlabels])

    O1 = layers(x, W_1, b_1)

    #second layer(output layer in this case)
    # W_2 = weight_variable([hiddenunits1, hiddenunits1])
    # b_2 = bias_variable([hiddenunits1])

    # O2 = tf.nn.relu(layers(O1, W_2, b_2))

    W_3 = weight_variable([hiddenunits1, numberlabels])
    b_3 = bias_variable([numberlabels])

    O3 = layers(O1, W_3, b_3)

    y_conv = O3

    #Train and Evaluate the Model

    # cost function to minimize (with L2 regularization)
    cross_entropy = tf.reduce_sum( -y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))-(1.0-y_)*tf.log((tf.clip_by_value(1-y_conv,1e-10,1.0))))  \
                     +lamb*(tf.nn.l2_loss(W_1)+tf.nn.l2_loss(W_3) )   

    #defining the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)  #0.0001 is learn_rate
    train_step = optimizer.minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) 

    #reading the data in the directory txt
    mnist = input_data.read_data_sets(numberlabels,
                                      lx,
                                      ly,
                                      './data/',
                                      one_hot=True)
 
    print(mnist)

    print('test.images.shape', mnist.test.images.shape)
    print('test.labels.shape', mnist.test.labels.shape)
    print(
        "xxxxxxxxxxxxxxxxxxxxx Training START xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        size)

    sess.run(tf.global_variables_initializer())  

    # training
    for i in range(1,trainstep+1):

        batch = mnist.train.next_batch(batch_size)

        if i % 2000 == 0:

            # batch_train = mnist.train.next_batch(batchsize_test)

            train_accuracy = sess.run(accuracy,
                                      feed_dict={
                                          x: batch[0],
                                          y_: batch[1]
                                      })
            print("step, train accuracy:", i, train_accuracy)

            # batch_test = mnist.test.next_batch(batchsize_test)

            # test_accuracy = sess.run(accuracy,
            #                          feed_dict={
            #                              x: batch_test[0],
            #                              y_: batch_test[1]
            #                          })
            # print("step, test accuracy:", i, test_accuracy)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})   

    print(
        "xxxxxxxxxxxxxxxxxxxxx Training Done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )

    print(
        "test accuracy",
        sess.run(accuracy,
                 feed_dict={
                     x: mnist.test.images,
                     y_: mnist.test.labels
                 }))

    # saver = tf.train.Saver()
    # save_path = saver.save(sess, "./model-saved.ckpt")
    # print("Model saved in path: %s" % save_path)

    print("xxxxxxxxxxxxxxxxxxxxx Plot Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    #producing data to get the plots we like
    #output of neural net

    # plist = ptrain = [0, 0.05, 0.10000, 0.133, 0.166, 0.20000, 0.233, 0.266, 0.30000, 0.333, 
    #                 0.366, 0.40000, 0.433, 0.466, 0.48, 0.50000, 0.533, 0.566, 0.58, 0.59,
    #                 0.60000,  0.61,   0.62,  0.63,  0.645,  0.66,  0.67,  0.68, 0.69,  0.70000, 
    #                 0.715, 0.733, 0.766, 0.80000, 0.833, 0.866, 0.90000, 0.933, 0.966, 1.00000]
    plist = ptrain = [0.0375 + x*0.0025 for x in range(31)]
    ptest = plist
    Ntemp = len(
        plist)  # number of different temperatures used in the simulation

    samples_per_T = int(mnist.test.num_examples / Ntemp)

    f = open('./plot/' + 'fcnoutlx' + str(lx) + '_' + str(ly) + '.dat', 'w')
    ii = 0
    for i in range(Ntemp):
      av=0.0
      for j in range(samples_per_T):
            batch=(mnist.test.images[ii,:].reshape((1,lx*ly)),mnist.test.labels[ii,:].reshape((1,numberlabels)))   
            res=sess.run(y_conv,feed_dict={x: batch[0], y_: batch[1]}) 
            av=av+res 
            ii=ii+1 
      av=av/samples_per_T
      print(plist[i],av[0,0],av[0,1])   
      f.write(str(plist[i])+' '+str(av[0,0])+' '+str(av[0,1])+"\n")

    f.close()
   

    f = open('./plot/' + 'fcnacclx' + str(lx) + '_' + str(lx) + '.dat', 'w')
    ii = 0
    # accuracy vs temperature
    for i in range(Ntemp):
        batch = (mnist.test.images[ii * samples_per_T:ii * samples_per_T +
                                   samples_per_T, :].reshape(
                                       samples_per_T, lx * ly),
                 mnist.test.labels[ii * samples_per_T:ii * samples_per_T +
                                   samples_per_T, :].reshape(
                                       (samples_per_T, numberlabels)))
        train_accuracy = sess.run(accuracy,
                                  feed_dict={
                                      x: batch[0],
                                      y_: batch[1]
                                  })
        ii = ii + 1
        f.write(str(ptest[i]) + ' ' + str(train_accuracy) + "\n")

    f.close()

end = time.time()
print('Running time: %s Seconds'%(end-start))