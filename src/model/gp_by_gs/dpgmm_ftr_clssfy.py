import pickle
import numpy as np
from sklearn import mixture
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm



input_path="../gp/cls_train_b.txt"
file_in=open(input_path,'r')
cls_train=pickle.load(file_in)

input_path="../gp/train_ftr_b.txt"
file_in=open(input_path,'r')
ftr_train=pickle.load(file_in)

input_path="../gp/test_ftr_b.txt"
file_in=open(input_path,'r')
ftr_test=pickle.load(file_in)

input_path="../gp/cls_test_b.txt"
file_in=open(input_path,'r')
cls_test=pickle.load(file_in)

ftr1=np.zeros([2*len(ftr_train),200],dtype="float")
for i in range(len(ftr_train)):
    #ftr1.append(ftr_train[i])
    print i
    tmp=ftr_train[i]
    ftr1[2*i]=tmp[0:200]
    ftr1[2*i+1]=tmp[200:]
#ftr1=np.array(ftr1,dtype='float')

#ftr2=np.abs(ftr_train[:,0:200]-ftr_train[:,200:])/(ftr_train[:,0:200]+ftr_train[:,200:])

# 

#ftr1_pca=PCA(n_components=20)
#ftr2_pca=PCA(n_components=20)

#ftr1=ftr1_pca.fit_transform(ftr1)
#ftr2=ftr2_pca.fit_transform(ftr2)



mydpgmm1=mixture.GMM(n_components=100,covariance_type="diag")
#mydpgmm2=mixture.GMM(n_components=8,covariance_type="diag")

mydpgmm1.fit(ftr1)
#mydpgmm2.fit(ftr2)

n1=mydpgmm1.n_components
#print n1
#n2=mydpgmm2.n_components

mean1=mydpgmm1.means_
#print mean1
#mean2=mydpgmm2.means_

precs1=mydpgmm1.covars_
#print precs1
#precs2=mydpgmm2.covars_

predict1=mydpgmm1.predict(ftr1)
#print predict1
#predict2=mydpgmm2.predict(ftr2)

predict_prob1=mydpgmm1.predict_proba(ftr1)
#predict_prob2=mydpgmm2.predict_proba(ftr2)

w1=[np.sum(predict1==i) for i in range(n1)]
#w2=[np.sum(predict2==i) for i in range(n2)]

ftrnew1=[]
ftrnew2=[]
pre1=[]
pre2=[]
for i in range(len(ftr1)):
    tmp=[]
    for j in range(n1):
         tmp_val=(ftr1[i]-mean1[j])/np.sqrt(precs1[j])
         tmp=np.hstack((w1[j]*np.sum(np.abs(tmp_val)),w1[j]*np.sum(np.square(tmp_val)),predict_prob1[i]))
    
    
    
    if np.mod(i,2)==0:     
        ftrnew1.append(tmp)
        pre1.append(predict1[i])
    else:
        pre2.append(predict1[i])
        ftrnew2.append(tmp)

ftrnew1=np.array(ftrnew1)
ftrnew2=np.array(ftrnew2)
#
#ftrnew1=np.concatenate((ftrnew1,np.hstack(pre1).T),axis=1)
#ftrnew2=np.concatenate((ftrnew2,np.hstack(pre1).T),axis=1)
ftrtrain=np.concatenate((ftrnew1,ftrnew2),axis=1)
    
#    tmp=[]
#    for j in range(n2):
#        tmp_val=(ftr2[i]-mean2[j])*precs2[j]
#        tmp=np.hstack((tmp,w2[j]*tmp_val))
#        tmp=np.hstack((tmp,w2[j]*np.square(tmp_val)))
#        tmp=np.hstack((tmp,predict_prob2[i]))
#    ftrnew2.append(tmp)




ftrtrain=np.array(ftrtrain)
#ftrnew2=np.array(ftrnew2)

#ftr1_new_pca=PCA(n_components=20)
#ftrnew1=ftr1_new_pca.fit_transform(ftrnew1)


clf1=RandomForestClassifier()
#clf2=RandomForestClassifier()
clf1=clf1.fit(ftrtrain,cls_train)
#clf2=clf2.fit(ftrnew2,cls_train)

#clf1=svm.SVC()
#clf1.fit(ftrnew1,cls_train)




print True

ftr_test1=ftr_test
#ftr_test2=np.abs(ftr_test[:,0:200]-ftr_test[:,200:])/(ftr_test[:,0:200]+ftr_test[:,200:])
#ftr_test1=ftr1_pca.transform(ftr_test1)
#ftr_test2=ftr2_pca.transform(ftr_test2)


ftr2=np.zeros([2*len(ftr_test),200],dtype="float")
for i in range(len(ftr_test)):
    #ftr1.append(ftr_train[i])
    
    tmp=ftr_test[i]
    ftr2[2*i]=tmp[0:200]
    ftr2[2*i+1]=tmp[200:]


predict2=mydpgmm1.predict(ftr2)
#print predict1
#predict2=mydpgmm2.predict(ftr2)

predict_prob2=mydpgmm1.predict_proba(ftr2)

ftrnewtest1=[]
ftrnewtest2=[]
pre1=[]
pre2=[]

for i in range(len(ftr2)):
    tmp=[]
    for j in range(n1):
        tmp_val=(ftr2[i]-mean1[j])/np.sqrt(precs1[j])
        tmp=np.hstack((w1[j]*np.sum(np.abs(tmp_val)),w1[j]*np.sum(np.square(tmp_val)),predict_prob2[i]))
        
    if np.mod(i,2)==0:
        ftrnewtest1.append(tmp)
        pre1.append(predict2[i])
    else:
        ftrnewtest2.append(tmp)
        pre2.append(predict2[i])
    
ftrnewtest1=np.array(ftrnewtest1)
ftrnewtest2=np.array(ftrnewtest2)        
#ftrnewtest1=np.concatenate((ftrnewtest1,np.array(pre1)),axis=1)
#ftrnewtest2=np.concatenate((ftrnewtest2,np.array(pre2)),axis=1)
ftrtest=np.concatenate((ftrnewtest1,ftrnewtest2),axis=1)
    
#for i in range(len(cls_test)): 
#    tmp=[]
#    for j in range(n2):
#        tmp_val=(ftr_test2[i]-mean2[j])/np.sqrt(precs1[j])
#        tmp=np.hstack((tmp,w2[j]*tmp_val))
#        tmp=np.hstack((tmp,w2[j]*np.square(tmp_val)))
#        tmp=np.hstack((tmp,predict_prob2[i]))
#    ftrnewtest2.append(tmp)


#ftrnewtest1=ftr1_new_pca.transform(ftrnewtest1)

y1=clf1.predict(ftrtest)
print np.sqrt(np.sum(np.square(y1-cls_test)))
#y2=clf2.predict(ftrnewtest2)
#print np.sqrt(np.sum(np.square(y2-cls_test)))
	
	

