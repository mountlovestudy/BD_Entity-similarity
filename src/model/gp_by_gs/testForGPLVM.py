# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 14:32:49 2014
test for GPLVM
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

num=500
numtest=500
dim=400


#ftr1=np.array(ftrall[1])
#ftr2=np.array(ftrall[2])
#ftr3=np.array(ftrall[3])
#ftr4=np.array(ftrall[4])

ftr1=np.zeros([num,dim])
ftr2=np.zeros([num,dim])
ftr3=np.zeros([num,dim])
ftr4=np.zeros([num,dim])

ftrt1=np.zeros([numtest,dim])
ftrt2=np.zeros([numtest,dim])
ftrt3=np.zeros([numtest,dim])
ftrt4=np.zeros([numtest,dim])




for i in range(num):
    ftr1[i,:]=uniform(ftrall[1][i])
    ftr2[i,:]=uniform(ftrall[2][i])
    ftr3[i,:]=uniform(ftrall[3][i])
    ftr4[i,:]=uniform(ftrall[4][i])
    
for i in range(1,numtest+1):
    ftrt1[i-1,:]=uniform(ftrall[1][-i])
    ftrt2[i-1,:]=uniform(ftrall[2][-i])
    ftrt3[i-1,:]=uniform(ftrall[3][-i])
    ftrt4[i-1,:]=uniform(ftrall[4][-i])
    
    
    
    
ftr=np.concatenate((ftr1,ftr2,ftr3,ftr4),axis=0)
ftrtest=np.concatenate((ftrt1,ftrt2,ftrt3,ftrt4),axis=0)
cls=np.hstack(np.concatenate((np.ones([2*num,1]),2*np.ones([2*num,1])),axis=0))
clstest=np.hstack(np.concatenate((np.ones([2*numtest,1]),2*np.ones([2*numtest,1])),axis=0))

del ftrall,ftr1,ftr2,ftr3,ftr4,ftrt1,ftrt2,ftrt3,ftrt4


myDGPLVM=DGPLVM(ftr,dim=20,cls=cls,delta=1e-3)
myDGPLVM.learn(20)

xx=myDGPLVM.predict(ftrtest.copy(),15,2)
plt.figure()
pb.scatter(myDGPLVM.GP.X[:,0],myDGPLVM.GP.X[:,1],c=cls-1,hold='on')
pb.scatter(xx[:,0],xx[:,1],c='g', marker='^')


y=myDGPLVM.classify(ftrtest.copy())
#
#
clf1=svm.SVC()
clf1.fit(myDGPLVM.GP.X,myDGPLVM.cls)
y1=clf1.predict(xx)




clf2=RandomForestClassifier()
#clf2=RandomForestClassifier()
clf2=clf2.fit(myDGPLVM.GP.X,myDGPLVM.cls)
y2=clf2.predict(xx)


clsreal=np.hstack(np.concatenate((np.ones([numtest,1]),2*np.ones([numtest,1]),3*np.ones([numtest,1]),4*np.ones([numtest,1])),axis=0))
ftr12=[]
ftr34=[]
clsreal12=[]
clsreal34=[]
for i in range(len(y2)):
    if y2[i]==1:
        ftr12.append(ftrtest[i])
        clsreal12.append(clsreal[i])        
    else:
        ftr34.append(ftrtest[i])
        clsreal34.append(clsreal[i])

ftr12=np.array(ftr12)
ftr34=np.array(ftr34)
clsreal12=np.array(clsreal12)
clsreal34=np.array(clsreal34)

##a="../../../src/model/gp/ftr12.txt"
##output=open(a,'w')
##pickle.dump(ftr12,output)
##output.close()
##a="../../../src/model/gp/ftr34.txt"
##output=open(a,'w')
##pickle.dump(ftr34,output)
##output.close()
##a="../../../src/model/gp/cls12.txt"
##output=open(a,'w')
##pickle.dump(clsreal12,output)
##output.close()
##a="../../../src/model/gp/cls34.txt"
##output=open(a,'w')
##pickle.dump(clsreal34,output)
##output.close()



#in_put=open("E:/BD/bddac-task1/src/model/gp/train_ftr_test500.txt","r")
#ftr=pickle.load(in_put)
#in_put.close()
#in_put=open("E:/BD/bddac-task1/src/model/gp/cls_train500.txt","r")
#cls=pickle.load(in_put)
#in_put.close()
#
##myGPLVM=GPLVM(ftr,dim=20)
##myGPLVM.learn(30)
#cls=cls/3+1
##myDGPLVM=DGPLVM(ftr,dim=20,cls=cls,delta=0.001)
#
##pca=PCA(n_components=50)
##ftrpca=pca.fit_transform(ftr[100:120,:])
#
#for i in range(len(ftr)):
#    tmp=ftr[i]
#    tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
#    ftr[i]=tmp
#
#
#myDGPLVM=DGPLVM(ftr,dim=20,cls=cls,delta=1e-3)
#myDGPLVM.learn(20)
##
##
##
#
#in_put=open("../gp/test_ftr_test.txt","r")
#ftrtest=pickle.load(in_put)
#in_put.close()
#in_put=open("../gp/cls_test.txt","r")
#clstest=pickle.load(in_put)
#in_put.close()
#
#
#for i in range(len(ftrtest)):
#    tmp=ftrtest[i]
#    tmp=(tmp-tmp.min())/(tmp.max()-tmp.min())
#    ftrtest[i]=tmp
#
#
#clstest=clstest/3+1
#
##ftrtestpca=pca.transform(ftrtest)
#
#
#xx=myDGPLVM.predict(ftrtest.copy(),15,3)
##
#plt.figure()
#pb.scatter(myDGPLVM.GP.X[:,0],myDGPLVM.GP.X[:,1],c=cls-1,hold='on')
#pb.scatter(xx[:,0],xx[:,1],c='g', marker='^')
###
##
#y=myDGPLVM.classify(ftrtest.copy())
##
##
#clf1=svm.SVC()
##X=myDGPLVM.GP.X
##X-=X.mean(1)
##X/=X.std(1)
#clf1.fit(myDGPLVM.GP.X,myDGPLVM.cls)
#y1=clf1.predict(xx)
###











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
