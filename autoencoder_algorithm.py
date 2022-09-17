import tensorflow as tf
import input_data
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

numberlabels = 2
hiddenunits = [2*512,64, 2, 64,512*2]

lamb = 0.001  # regularization parameter
batchsize_test = 1000
learning_rate = 0.0001
batch_size = 32
trainstep = (2500 * 10 // batch_size)*20
kernel_size = 3
kernel_size_1 = 3
kernel_size_2 = 3
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

def layers_R(x, W, b):
    return tf.nn.relu(tf.matmul(x, W)+b)

def layers_N(x, W, b):
    return tf.matmul(x, W)+b

def conv2d(x,W,b):
    return tf.nn.relu(tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b)

def conv2d_S(x,W,b):
    return tf.nn.sigmoid(tf.nn.conv2d(x,W,[1,1,1,1],padding='SAME')+b)

def max_pool_2(x):
    return tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')

sess = tf.Session()

LENGTH_GET = [40]
STEP = [[1,41]]  
# LENGTH_GET = [8,16,32,48,64]
# STEP = [[1,9],[1,17],[1,33],[1,49],[1,65]] 

for size in range(len(LENGTH_GET)):
    # Size
    lx = LENGTH_GET[size]
    # Time
    ly = STEP[size][1] - STEP[size][0]
    # defining the model
        #x is input_data, y_ is the label
    x = tf.placeholder("float", shape=[None, lx * ly])   
    y_ = tf.placeholder("float", shape=[None, lx * ly])

    xinput_re = tf.reshape(x,[-1,lx,ly,1])
    # encoder
    W_1 = weight_variable([kernel_size_1,kernel_size_2,1,16])
    b_1 = bias_variable([16])
    O1 = conv2d_S(xinput_re, W_1, b_1)  #lx*lx
    # 100 > 50
    W_2 = weight_variable([kernel_size_1, kernel_size_2, 16, 8])
    b_2 = bias_variable([8])
    O2 = conv2d_S(max_pool_2(O1), W_2, b_2)#(lx//2 ) *(lx//2)

    W_21 = weight_variable([kernel_size_1, kernel_size_2, 8, 8])
    b_21 = bias_variable([8])
    O21 = conv2d_S(max_pool_2(O2), W_21, b_21) #(lx//4 ) *(lx//4)
    O2_re = tf.reshape(max_pool_2(O21),[-1,(lx//8 )*(ly//8)*8])  #(lx//8 ) *(lx//8)
    # 50 > 2
    W_3 = weight_variable([(lx//8 )*(ly//8)*8, 1])
    b_3 = bias_variable([1])
    O3 = layers_N(O2_re, W_3, b_3)
    # decoder 
    W_4 = weight_variable([1, (lx//8 )*(ly//8)*8])
    b_4 = bias_variable([(lx//8 )*(ly//8)*8])
    O4 = layers(O3, W_4, b_4)
    O4_re = tf.reshape(O4,[-1,lx//8,ly//8,8])
    # 50 > 100
    W_5 = weight_variable([kernel_size_1,kernel_size_2,8, 8])
    b_5 = bias_variable([8])
    O5 = conv2d_S(tf.image.resize_images(O4_re,(lx//4,ly//4),method=1), W_5, b_5)
    
    W_51 = weight_variable([kernel_size_1,kernel_size_2,8, 16])
    b_51 = bias_variable([16])
    O51 = conv2d_S(tf.image.resize_images(O4_re,(lx//2,ly//2),method=1), W_51, b_51)
    # 100 > lx*ly
    W_6 = weight_variable([kernel_size_1,kernel_size_2,16,16])
    b_6 = bias_variable([16])
    O6 = conv2d_S(tf.image.resize_images(O51,(lx,ly),method=1), W_6, b_6)

    W_7 = weight_variable([kernel_size_1,kernel_size_2,16,1])
    b_7 = bias_variable([1])
    O7 = conv2d_S(O6, W_7, b_7)

    y_conv = tf.reshape(O7,[-1,lx*ly])

    #Train and Evaluate the Model
    # cost function to minimize (with L2 regularization)
    cross_entropy = tf.reduce_mean( -y_*tf.log(tf.clip_by_value(y_conv,1e-10,1.0))-(1.0-y_)*tf.log((tf.clip_by_value(1-y_conv,1e-10,1.0))))  
    #defining the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)  #0.0001 is learn_rate
    train_step = optimizer.minimize(cross_entropy)

    
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

            train_loss = sess.run(cross_entropy,
                                      feed_dict={
                                          x: batch[0],
                                          y_: batch[0]
                                      })
            print("step, train loss:", i, train_loss)

            
       
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[0]})   

    print(
        "xxxxxxxxxxxxxxxxxxxxx Training Done xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    )

    print(
        "test loss",
        sess.run(cross_entropy,
                 feed_dict={
                     x: mnist.test.images,
                     y_: mnist.test.images
                 }))

    print("xxxxxxxxxxxxxxxxxxxxx Plot Data xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

    #producing data to get the plots we like
    #output of neural net
    # plist = ptrain = [0.1 + x*0.1 for x in range(10)]
    # plist = ptrain = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    plist = ptrain = [0.0 + x*0.01 for x in range(31)]
    # plist = ptrain = [0.0 + x*0.025 for x in range(41)]
    # plist = ptrain = [0.0 + x*0.005 for x in range(41)]
    # plist = ptrain = [0.1, 0.6447, 0.9]
    ptest = plist
    Ntemp = len(plist)  # number of different temperatures used in the simulation
    print(Ntemp)
    samples_per_T = int(mnist.test.num_examples / Ntemp)

    f = open('./plot/' + 'nnoutlxalldensityd02' + str(lx) + '_' + str(ly) + '.dat', 'w')
    ii = 0
    av_T =[]
    av_x_ALL= []
    av_y_ALL= []
    for i in range(Ntemp):
        # av_z = []
        av=0.0
        for j in range(samples_per_T):
           
            res=sess.run(O3,feed_dict={x: [mnist.test.images[ii]]})  
            # print(res)
            # av.append(res[0]) 
            av=av+res 
            ii +=1
            # if ii == 25000:
            #     # print(mnist.train.images[ii,:])
            #     plt.imshow(mnist.train.images[ii,:].reshape((lx,ly)))
            #     plt.show()
            #     plt.imshow(sess.run(O7,feed_dict={x: batch[0], y_: batch[0]}).reshape((lx,ly)))
            #     plt.show()
        av=av/samples_per_T
        # av_z_ALL.append(av_z)
        print(av)
    # for i in range(len(plist)):
        plt.scatter(plist[i],av[0][0],label="{}".format(plist[i]))
        f.write(str(plist[i])+' '+str(av[0,0])+' '+"\n")
    plt.xlabel('${p}$',fontsize=20)
    plt.ylabel('${h^*}$',fontsize=20)
    # plt.figure(figsize=(8,6))
    # plt.title('A single hidden neuron activation as a function of ${p}$')
    plt.savefig('ae_pc_alldensity.pdf')
    plt.show()