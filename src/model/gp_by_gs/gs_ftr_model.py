
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 11:42:31 2014
the Gaussian feature based on the DGPLVM and mutual information
@author: mountain
"""

from GPLVM_test import DGPLVM
import numpy as np
from scg import SCG

class GS_ftr_model:
    def __init__(self,Y_tar,cls_tar,Y_src,cls_src,beta,dim=20,delta=1e-1):
        #beta is the coefficent of the mutual information
        #delta is the para in X_prior
        #dim is the dimension of the latent variable X
        self.Y_tar=Y_tar
        self.cls_tar=cls_tar
        self.Y_src=Y_src
        self.cls_src=cls_src
        self.beta=beta
        self.DGPLVM_tar=DGPLVM(Y_tar,dim,cls_tar,delta)
        self.DGPLVM_src=DGPLVM(Y_src,dim,cls_src,delta)
        self.Y=np.concatenate((self.Y_tar, self.Y_src), axis=0)
        self.cls=np.concatenate((self.cls_tar,self.cls_src))
        self.DGPLVM_all=DGPLVM(self.Y,dim,self.cls,delta)
        
    
    def poster_hyper(self):        
        self.poster_tar=np.exp(self.DGPLVM_tar.GP.marginal())+np.exp(self.DGPLVM_tar.GP.hyper_prior())        
        self.poster_all=np.exp(self.DGPLVM_all.GP.marginal())+np.exp(self.DGPLVM_all.GP.hyper_prior())
    
    
    
    def ll_hyper(self,params=None):
        self.DGPLVM_tar.GP.set_params(params)
        self.DGPLVM_all.GP.set_params(params)
        self.poster_hyper()
        return self.DGPLVM_tar.GP.ll(params=params)*(1-self.beta*self.poster_tar)+\
        self.beta*self.poster_all*(self.DGPLVM_all.GP.ll(params=params)-self.DGPLVM_src.GP.ll(params=params))
        
    
    def ll_hyper_grad(self,params=None):
        self.DGPLVM_tar.GP.set_params(params)
        self.DGPLVM_all.GP.set_params(params)
        self.poster_hyper()
        return -(self.beta*(-self.DGPLVM_tar.GP.ll(params)+1)*self.poster_tar-1)*self.DGPLVM_tar.GP.ll_grad(params)\
        -self.beta*self.poster_all*self.DGPLVM_src.GP.ll_grad(params)\
        -self.beta*(self.DGPLVM_all.GP.ll(params)-self.DGPLVM_src.GP.ll(params)-1)*self.DGPLVM_all.GP.ll_grad(params)*self.poster_all
        
        
    def optimise_GP_kernel(self,iters=1000):
        """Optimise the marginal likelihood. work with the log of beta - fmin works better that way.  """
        
        new_params=SCG(self.ll_hyper,self.ll_hyper_grad,np.hstack((self.DGPLVM_tar.GP.kernel.get_params(), np.log(self.DGPLVM_tar.GP.beta))),maxiters=iters,display=True,func_flg=0)
        #gtol=1e-10,epsilon=1e-10,
#        new_params = fmin_cg(self.ll,np.hstack((self.kernel.get_params(), np.log(self.beta))),fprime=self.ll_grad,maxiter=iters,gtol=1e-10,disp=False)        
        self.DGPLVM_src.GP.set_params(new_params)
        self.DGPLVM_tar.GP.set_params(new_params)
        self.DGPLVM_all.GP.set_params(new_params)
        
        
    def poster_data(self,data_type):
        if data_type=="tar":
            self.poster_data_tar=np.exp(self.DGPLVM_tar.GP.marginal())+np.exp(self.DGPLVM_tar.x_prior())            
        elif data_type=="all":
            self.poster_data_all=np.exp(self.DGPLVM_all.GP.marginal())+np.exp(self.DGPLVM_all.x_prior())
            
    def ll(self,xx,i,xx_l,data_type):
        if data_type=="tar":
            self.DGPLVM_tar.GP.X[i]=xx
            self.DGPLVM_tar.GP.update()
            self.poster_data("tar")
            self.DGPLVM_all.GP.X[i]=xx
            self.DGPLVM_all.GP.update()
            self.poster_data("all")
            return self.DGPLVM_tar.ll(xx,i,xx_l,0)*(1-self.beta*self.poster_data_tar)+\
            self.beta*self.poster_data_all*(self.DGPLVM_all.ll(xx,i,xx_l,0)+self.DGPLVM_src.GP.marginal_value+self.DGPLVM_src.x_prior_value)
             
        elif data_type=="src":
            self.DGPLVM_src.GP.X[i]=xx
            self.DGPLVM_src.GP.update()
            self.DGPLVM_all.GP.X[i+self.DGPLVM_tar.N]=xx
            self.DGPLVM_all.GP.update()
            self.poster_data("all")
#            return (-self.DGPLVM_tar.GP.marginal()-self.DGPLVM_tar.x_prior())*(1-self.beta*self.poster_data_tar)+\
#            self.beta*self.poster_data_all*(self.DGPLVM_all.ll(xx,i+self.DGPLVM_tar.N,xx_l)-self.DGPLVM_src.ll(xx,i,xx_l))
            return self.beta*self.poster_data_all*(self.DGPLVM_all.ll(xx,i+self.DGPLVM_tar.N,xx_l,0)-self.DGPLVM_src.ll(xx,i,xx_l))
       
    def ll_grad(self,xx,i,xx_l,data_type):
        if data_type=="tar":
            self.DGPLVM_tar.GP.X[i]=xx
            self.DGPLVM_tar.GP.update()
            self.poster_data("tar")
            self.DGPLVM_all.GP.X[i]=xx
            self.DGPLVM_all.GP.update()
            self.poster_data("all")
            return -(self.beta*(-self.DGPLVM_tar.ll(xx,i,xx_l,0)+1)*self.poster_data_tar-1)*self.DGPLVM_tar.ll_grad(xx,i,xx_l,0)\
            -self.beta*(self.DGPLVM_all.ll(xx,i,xx_l,0)-1+self.DGPLVM_src.GP.marginal_value+self.DGPLVM_src.x_prior_value)*self.poster_data_all*self.DGPLVM_all.ll_grad(xx,i,xx_l,0)
            
        if data_type=="src":
            self.DGPLVM_all.GP.X[i+self.DGPLVM_tar.N]=xx
            self.DGPLVM_all.GP.update()
            self.poster_data("all")
            
            #self.poster_data("tar")
            #return (-self.DGPLVM_tar.GP.marginal()-self.DGPLVM_tar.x_prior())*(1-self.beta*self.poster_data_tar)-self.beta*self.poster_data_all*self.DGPLVM_src.ll_grad(xx,i,xx_l)\
            #-self.beta*(self.DGPLVM_all.ll(xx,i+self.DGPLVM_tar.N,xx_l)-self.DGPLVM_src.ll(xx,i,xx_l)-1)*self.poster_data_all*self.DGPLVM_all.ll_grad(xx,i+self.DGPLVM_tar.N,xx_l)
            return   -self.beta*self.poster_data_all*self.DGPLVM_src.ll_grad(xx,i,xx_l)\
            -self.beta*(self.DGPLVM_all.ll(xx,i+self.DGPLVM_tar.N,xx_l,0)-self.DGPLVM_src.ll(xx,i,xx_l)-1)*self.poster_data_all*self.DGPLVM_all.ll_grad(xx,i+self.DGPLVM_tar.N,xx_l,0)          
            
    def optimise_latents(self):
        xtemp=np.zeros(self.DGPLVM_tar.GP.X.shape)
        xtemp_src=np.zeros(self.DGPLVM_src.GP.X.shape)
        
        self.DGPLVM_src.GP.marginal()
        self.DGPLVM_src.x_prior()
        for i,yy in enumerate(self.DGPLVM_tar.GP.Y):
            original_x = self.DGPLVM_tar.GP.X[i].copy()
            xx_l=self.DGPLVM_tar.cls[i]
            #xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,gtol=1e-10,disp=True,args=(i,xx_l))
            xopt=SCG(self.ll,self.ll_grad,self.DGPLVM_tar.GP.X[i],optargs=(i,xx_l,"tar"),display=False)
            self.DGPLVM_tar.GP.X[i] = original_x
            xtemp[i] = xopt
        
        
        for i,yy in enumerate(self.DGPLVM_src.GP.Y):
            original_x = self.DGPLVM_src.GP.X[i].copy()
            xx_l=self.DGPLVM_src.cls[i]
            #xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,gtol=1e-10,disp=True,args=(i,xx_l))
            xopt=SCG(self.ll,self.ll_grad,self.DGPLVM_src.GP.X[i],optargs=(i,xx_l,"src"),display=False)
            self.DGPLVM_src.GP.X[i] = original_x
            xtemp_src[i] = xopt
        
        self.DGPLVM_tar.GP.X=xtemp.copy()
        self.DGPLVM_src.GP.X=xtemp_src.copy()
        
    def learn(self,niters):
        for i in range(niters):
            self.optimise_latents()
            self.optimise_GP_kernel()
            
    def predict(self,ynew,nhidden=5,mlp_alpha=2):
        return self.DGPLVM_tar.predict(ynew,nhidden,mlp_alpha)
        
        
            
    
    
    

