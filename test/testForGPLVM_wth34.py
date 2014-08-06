# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 13:14:49 2014
test for 12
@author: mountain
"""


##
from GPLVM import DGPLVM,GPLVM
import pickle
import pylab as pb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def uniform(tmp):
    tmp=np.array(tmp)
    return (tmp-tmp.min())/(tmp.max()-tmp.min())

in_put=open("E:/BD/bddac-task1/src/model/gp/train_ftr_differ_lbl.txt","r")
ftrall=pickle.load(in_put)
in_put.close()


num=25
numtest=50
dim=400
lbl1=3
lbl2=4

ftr1=np.zeros([num,dim])
ftr2=np.zeros([num,dim])


ftrt1=np.zeros([numtest,dim])
ftrt2=np.zeros([numtest,dim])





for i in range(num):
    ftr1[i,:]=uniform(ftrall[lbl1][i])
    ftr2[i,:]=uniform(ftrall[lbl2][i])
    
for i in range(1,numtest+1):
    ftrt1[i-1,:]=uniform(ftrall[lbl1][-i])
    ftrt2[i-1,:]=uniform(ftrall[lbl2][-i])
    
    
    
    
ftr=np.concatenate((ftr1,ftr2),axis=0)
ftrtest=np.concatenate((ftrt1,ftrt2),axis=0)
cls=np.hstack(np.concatenate((np.ones([num,1]),2*np.ones([num,1])),axis=0))
clstest=np.hstack(np.concatenate((np.ones([numtest,1]),2*np.ones([numtest,1])),axis=0))


myDGPLVM34=DGPLVM(ftr,dim=20,cls=cls,delta=1e-3)
myDGPLVM34.learn(25)




for i in range(len(ftrtest)):
    tmp=ftrtest[i]
    tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
    ftrtest[i]=tmp



xx=myDGPLVM34.predict(ftrtest.copy(),10,2)
#
plt.figure()
pb.scatter(myDGPLVM34.GP.X[:,0],myDGPLVM34.GP.X[:,1],c=cls-1,hold='on')
pb.scatter(xx[:,0],xx[:,1],c='g', marker='^')
#

clf31=svm.SVC()
clf31.fit(myDGPLVM34.GP.X,myDGPLVM34.cls)
y31=clf31.predict(xx)

clf32=RandomForestClassifier()
clf32=clf32.fit(myDGPLVM34.GP.X,myDGPLVM34.cls)
y32=clf32.predict(xx)









