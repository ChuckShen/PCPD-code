import json
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.utils.extmath import svd_flip
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc,rcParams
import matplotlib
import random
import time
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
start=time.time()

# font = {'size':30}
# matplotlib.rc('font', **font)

# ------------------------------------------------------
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

run_time = 100
d = 0.1
pc = 0.10688
dl = d/2
dr = d
# ptrain = [0.1, 0.6447, 0.9]
ptrain = [0.0 + x*0.01 for x in range(31)]
# ptrain = [0.0 + x*0.01 for x in range(31)]
ptest = ptrain
plist = ptest
z = 1.

LENGTH = 40
# TIME = np.math.ceil(LENGTH**z)
# print(TIME)
TIME = LENGTH-1
print(TIME)

a_test = []
for p in ptest:
    ann = dr + p*(1-dr)
    cl = ann + (1-p)*(1-dr)/2
    for icyc in range(0, run_time):
        # vector = []
        vector_LT = []
        V_pro = np.ones(LENGTH)
        vector_LT.extend(V_pro)    
        for i in range(TIME):
            V_pro = Update(V_pro, dl, dr, ann, cl)
            vector_LT.extend(V_pro)
        a_test.append(vector_LT)

aa_test = []
for p in ptest:
    for icyc in range(0, run_time):
        array = []
        array = p
        aa_test.append(array)
# print(aa_test)

X = np.array(a_test)
print(X.shape)
y = np.array(aa_test)
print(y)
print(y.shape)
# target_names = iris.target_names
# pca = PCA(n_components=2)
pca = PCA(n_components = 10, svd_solver='full')
# print(pca)
pca.fit(X)
# print(pca.fit(X))
X_reduction = pca.transform(X)
namda = pca.explained_variance_ratio_
print(namda) 
print(namda.shape)
# print(pca.singular_values_)

# print(X_reduction)
# print(X_reduction.shape)
# print (pca.n_components_)
# print(X_reduction[100,0])

#--------------------------------------------------------------------------
# To make plots pretty
golden_size = lambda width: (width, 2. * width / (1 + np.sqrt(5)))
font1 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 60,
         }
plt.rc('font',**{'size':60})
# rc('text',usetex=True)
# rcParams.update({'font.size': 16})
fig, ax = plt.subplots(1,figsize=golden_size(26))
order = [1 + x*1 for x in range(10)]
print(order)
ax.plot(order[0:], namda[0:], color='red')
sc = ax.scatter(order[0:], namda[0:], s=60,marker = "x", color='blue')
ax.set_xlabel(u'${\ell}$',fontsize=60)
ax.set_ylabel(u'$ \~{\lambda}_{\ell} $',fontsize=60) 
y=[pow(10,i) for i in range(-3,0)]
plt.yscale('log')
ax.set_ylim(10**-3, 10**0)
plt.xlim(0,10.5)
# plt.ylim(-pow(10,-0.8),pow(10,0))
ax.tick_params(axis='both',which='both',direction='in',top=True,right=True,labeltop=False,labelright = False)
plt.tick_params(which='major',width=4,length=32)
plt.tick_params(which='minor',width=4,length=16)

xmajorLocator = MultipleLocator(2) 
xmajorFormatter = FormatStrFormatter('%.0f') 
xminorLocator   = MultipleLocator(1) 
# ymajorLocator = MultipleLocator(0.5)
# ymajorFormatter = FormatStrFormatter('%.1f') 
# yminorLocator   = MultipleLocator(0.1)   

ax.xaxis.set_major_locator(xmajorLocator)  
ax.xaxis.set_major_formatter(xmajorFormatter)  
ax.xaxis.set_minor_locator(xminorLocator) 

# ax.yaxis.set_major_locator(ymajorLocator)  
# ax.yaxis.set_major_formatter(ymajorFormatter)  
# ax.yaxis.set_minor_locator(yminorLocator)  

# plt.colorbar(sc, label='$0.25\\times$Temperature')
# plt.colorbar(sc)
# plt.legend(sc)
plt.savefig('pcpd1d_varianceratio.pdf')

end=time.time()
print('Running time: %s Seconds'%(end-start))

plt.show()