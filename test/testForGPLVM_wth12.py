# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 14:32:49 2014
test for GPLVM 34
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
lbl1=1
lbl2=2

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


myDGPLVM12=DGPLVM(ftr,dim=20,cls=cls,delta=1e-3)
myDGPLVM12.learn(25)




for i in range(len(ftrtest)):
    tmp=ftrtest[i]
    tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
    ftrtest[i]=tmp



xx=myDGPLVM12.predict(ftrtest.copy(),10,0.5)
#
plt.figure()
pb.scatter(myDGPLVM12.GP.X[:,0],myDGPLVM12.GP.X[:,1],c=cls-1,hold='on')
pb.scatter(xx[:,0],xx[:,1],c='g', marker='^')
#

clf11=svm.SVC()
clf11.fit(myDGPLVM12.GP.X,myDGPLVM12.cls)
y11=clf11.predict(xx)

clf12=RandomForestClassifier()
clf12=clf12.fit(myDGPLVM12.GP.X,myDGPLVM12.cls)
y12=clf12.predict(xx)

#y=myDGPLVM12.classify(ftrtest.copy())
##
#for i in range(len(y)):
#    if y[i]>=0.5:
#        y[i]=1
#    else:
#        y[i]=2








