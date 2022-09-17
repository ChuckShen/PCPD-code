import numpy as np
import random
import matplotlib.pyplot as plt

# -----------generate-------------------------------------------
def Update(V_pro, dl, dr, ann, cl):
    L = np.shape(V_pro)[0]
    V_aft = np.copy(V_pro)
    for i in range(L):
        h = (i - 1 + L) % L
        j = (i + 1) % L
        k = (i + 2) % L
        l = (i + 3) % L

        prob = random.random()
        if prob<dl: #diff to right
            if V_pro[j] == 1 and  V_pro[k] ==0:
                V_aft[j] =0
                V_aft[k] =1

        elif prob<dr: #diff to left
            if V_pro[j] == 1 and  V_pro[i] ==0:
                V_aft[j] =0
                V_aft[i] =1

        elif prob < ann: 
            if V_pro[j] ==1 and V_pro[k] ==1:
                V_aft[j] =0
                V_aft[k] =0

        elif prob < cl: #create at right
            if V_pro[j] ==1 and V_pro[k] ==1 and V_pro[l] ==0:
                V_aft[l] =1

        else: #create at left
            if V_pro[i] ==1 and V_pro[j] ==1 and V_pro[h] ==0:
                V_aft[h] =1
    return V_aft


ncyc_train = 2500
ncyc_test = 500

# ptrain = [0.1 + x*0.1 for x in range(10)]
# ptrain = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
ptrain = [0.0375 + x*0.0025 for x in range(31)]
# print(ptrain)
# ptrain = [0.0 + x*0.025 for x in range(41)]
# ptrain = [0.1, 0.9]
ptest = ptrain

d = 0.0
pc = 0.077092
# L = 1000
# T = 1000

dl = d/2
dr = d


# ------------------------------------------------------
z = 1.

LENGTH_GET = [80]
STEP = [[1,81]] 
for size in range(len(LENGTH_GET)):
    # Size
    LENGTH = LENGTH_GET[size]
    # Time
    TIME = np.math.ceil(LENGTH**z)
    a_test = []
    for p in ptest:
        ann = dr + p*(1-dr)
        cl = ann + (1-p)*(1-dr)/2
        for icyc in range(0, ncyc_test):
            # vector = []
            vector_LT = []
            V_pro = np.ones(LENGTH)
            vector_LT.append(V_pro)    
            for i in range(TIME):
                V_pro = Update(V_pro, dl, dr, ann, cl)
                vector_LT.append(V_pro)
                # print(vector_LT)
            # if icyc == 99:
            #     plt.imshow(vector_LT)
            #     plt.show()
            # if icyc ==1800:
            #     print(vector_LT)
            a_test.append(vector_LT[STEP[size][0] - 1:STEP[size][1] - 1])
    # print(a_test)

    np.save('./data/' + 'xtest_' + str(LENGTH) + '_' + str(TIME) + '.npy',
            a_test)

    fy = open('./data/' + 'ytest_' + str(LENGTH) + '_' + str(TIME) + '.dat',
              "w")
    for p in ptest:
        for icyc in range(0, ncyc_test):
            if (p < pc):
                fy.write("%5d \n" % 1)
            else:
                fy.write("%5d \n" % 0)
    fy.close()

    b_test = []
    for p in ptrain:
        ann = dr + p*(1-dr)
        cl = ann + (1-p)*(1-dr)/2
        for icyc in range(0, ncyc_train):
            # vector = []
            vector_LT = []
            V_pro = np.ones(LENGTH)
            vector_LT.append(V_pro) 
            for i in range(TIME):
                V_pro = Update(V_pro, dl, dr, ann, cl)
                vector_LT.append(V_pro)

            b_test.append(vector_LT[STEP[size][0] - 1:STEP[size][1] - 1])

    np.save('./data/' + 'xtrain_' + str(LENGTH) + '_' + str(TIME) + '.npy',
            b_test)

    fy = open('./data/' + 'ytrain_' + str(LENGTH) + '_' + str(TIME) + '.dat',
              "w")
    for p in ptrain:
        for icyc in range(0, ncyc_train):
            if (p < pc):
                fy.write("%5d \n" % 1)
            else:
                fy.write("%5d \n" % 0)
    fy.close()