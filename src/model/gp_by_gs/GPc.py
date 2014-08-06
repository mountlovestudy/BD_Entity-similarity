# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 21:41:27 2014
GPC with the laplace model
@author: mountain
"""

from scipy import special
import numpy as np
#from kernels import kernels
from scipy import optimize

class GPc:
    def __init__(self,K,cls):
        self.K=K
        self.cls=cls
        self.fnew=np.zeros(self.cls.shape)
        
    def ll(self,f):
        return -np.sum(np.log(special.ndtr(self.cls*f)))+0.5*np.dot(np.dot(self.fnew.T,np.linalg.inv(self.K)),f)
    
    
    def get_f(self):
        fnew=optimize.fmin_bfgs(self.ll,self.fnew)
        self.fnew=fnew
    def gs_prob(self,x):
        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*np.square(x))
        
    def get_var(self):
        f_prob=self.gs_prob(self.fnew)
        phai=special.ndtr(self.cls*self.fnew)
        return np.square(f_prob/phai)+self.cls*self.fnew*f_prob/phai
        
    def cal_f_w(self,maxiter=100):
        self.get_f()
        self.W=np.diag(self.get_var())
        #return self.f,W
    
#    def delt_f(self,f):
#        return self.cls*self.gs_prob(f)/special.ndtr(self.cls*f)
#    
#    def gs_prob(self,x):
#        return 1/np.sqrt(2*np.pi)*np.exp(-0.5*np.square(x))
#    
#    def cal_f_w(self,maxiter=10000):
#        fnew=np.zeros(self.cls.shape)
#        iter=0
#        while 1:
#            f=fnew
#            
#            tmp=np.zeros(fnew.shape)
#            for i in range(len(tmp)):
#                if special.ndtr(self.cls[i]*f[i])==0.0:
#                    tmp[i]=1
#                else:
#                    tmp[i]=self.gs_prob(f[i])/special.ndtr(self.cls[i]*f[i])            
#            
#            #tmp=self.gs_prob(f)/special.ndtr(self.cls*f)
#            
#            W=np.diag(np.square(tmp)+self.cls*f*tmp/np.sqrt(2*np.pi))
#            L=np.linalg.cholesky(np.eye(len(self.cls))+np.dot(np.dot(np.sqrt(W),self.K),np.sqrt(W)))
#            b=np.dot(W,f)+self.delt_f(f)
#            a=b-np.dot(np.linalg.inv(np.dot(np.sqrt(W),L.T)),np.dot(np.linalg.inv(L),np.dot(np.dot(np.sqrt(W),self.K),b)))
#            fnew=np.dot(self.K,a)
#            iter=iter+1
#            print np.sqrt(np.sum(np.square(fnew-f)))
#            if np.sqrt(np.sum(np.square(fnew-f)))<1e-6 or iter>maxiter:
#                break
#        
#        
#        tmp=self.gs_prob(fnew)/special.ndtr(self.cls*fnew)
#        W=np.diag(np.square(tmp)+self.cls*fnew*tmp)
#        L=np.linalg.cholesky(np.eye(len(self.cls))+np.dot(np.dot(np.sqrt(W),self.K),np.sqrt(W)))
#        self.fnew=fnew
#        self.W=W
#        self.L=L
#        self.delta=self.delt_f(fnew)
        

        
        
        
        
        
        
        
        
    
        
    
    
