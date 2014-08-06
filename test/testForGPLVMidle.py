# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 14:32:49 2014
test for GPLVM
@author: mountain
"""
##
from gs_dgplvm import DGPLVM,GPLVM
import pickle
import pylab as pb
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
#
#train_ftr="E:/BD/bddac-task1/src/model/gp/train_ftr_test.json"
#lsi_file="E:/BD/bddac-task1/result/item_ftr_aft_lsi.txt"
#cls_ftr="E:/BD/bddac-task1/src/model/gp/cls_train.json"
#ftr,cls=get_lsi_train(train_file,item2num_file,lsi_file)



in_put=open("E:/BD/bddac-task1/src/model/gp/train_ftr_test.txt","r")
ftr=pickle.load(in_put)
in_put.close()
in_put=open("E:/BD/bddac-task1/src/model/gp/cls_train.txt","r")
cls=pickle.load(in_put)
in_put.close()

#myGPLVM=GPLVM(ftr,dim=20)
#myGPLVM.learn(30)
cls=cls/3+1
#myDGPLVM=DGPLVM(ftr,dim=20,cls=cls,delta=1e-8)

pca=PCA(n_components=50)
ftr=pca.fit_transform(ftr)

myDGPLVM=DGPLVM(ftr,xdim=10,cls=cls,delta=0.1)
myDGPLVM.learn(100)
#
#
#

in_put=open("../gp/test_ftr_test.txt","r")
ftrtest=pickle.load(in_put)
in_put.close()
in_put=open("../gp/cls_test.txt","r")
clstest=pickle.load(in_put)
in_put.close()

clstest=clstest/3+1

ftrtestpca=pca.transform(ftrtest)


xx=myDGPLVM.MLP.forward(ftrtestpca)
#
plt.figure()
pb.scatter(myDGPLVM.GP.X[:,0],myDGPLVM.GP.X[:,1],c=cls-1,hold='on')

pb.scatter(xx[:,0],xx[:,1],c='g', marker='^')
##
#
#y=myDGPLVM.classify(ftrtest)
#
##
clf1=svm.SVC()
clf1.fit(myDGPLVM.GP.X,myDGPLVM.cls)
y1=clf1.predict(xx)
##


#tt=X[727:732]
#ttnew=myDGPLVM.predict(ynew=tt)
#pb.scatter(ttnew[:,0],ttnew[:,1],c='y', marker='^')

#import GPy
#import pylab as pb
#from GPLVM import DGPLVM,GPLVM
#import matplotlib.pyplot as plt
#import numpy as np
#from sklearn import svm
#
#
#
#data = GPy.util.datasets.oil()
#X = data['X']
#Xtest = data['Xtest']
#Y = data['Y'][:, 0:1]
#Ytest = data['Ytest'][:, 0:1]
#Y[Y.flatten()==-1] = 0
#Ytest[Ytest.flatten()==-1] = 0
#Y=np.array(Y,dtype=int)
#Y=np.hstack(Y)
#Y=Y+1
#
#myDGPLVM=DGPLVM(X[180:230,:],dim=2,cls=Y[180:230],delta=1e-1)
#myDGPLVM.learn(40)
#
#
#tt=X[13:22]
#ttnew=myDGPLVM.predict_x(ynew=tt)
#plt.figure()
#pb.scatter(myDGPLVM.GP.X[:,0],myDGPLVM.GP.X[:,1],c=Y[180:230]-1)
#pb.scatter(ttnew[:,0],ttnew[:,1],c='y', marker='^')
#y=myDGPLVM.classify(tt)
#
#
#
#clf1=svm.SVC()
#clf1.fit(myDGPLVM.GP.X,myDGPLVM.cls)
#y1=clf1.predict(ttnew)

#import GPy
#import pylab as pb
#import matplotlib.pyplot as plt
#from gs_ftr_model import GS_ftr_model
#
#
#data = GPy.util.datasets.oil()
#X = data['X']
#Xtest = data['Xtest']
#Y = data['Y'][:, 0:1]
#Ytest = data['Ytest'][:, 0:1]
#Y[Y.flatten()==-1] = 0
#Ytest[Ytest.flatten()==-1] = 0
#Y=np.array(Y,dtype=int)
#Y=np.hstack(Y)
#Y=Y+1
#
##Y_tar,cls_tar,Y_src,cls_src,beta,dim=20,delta=1e-1
#mygs=GS_ftr_model(X[180:280,:],Y[180:280],X[350:400],Y[350:400],1,dim=2,delta=1e-1)
#mygs.learn(30)


#import pickle
#
#input_path="../../../result/gp_test_label.txt"
#file_in=open(input_path,'r')
#
#tt=pickle.load(file_in)
