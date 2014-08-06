# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 23:05:09 2014
The Gaussian Process
refered to the work by James Hensman in github https://github.com/jameshensman/pythonGPLVM
@author: mountain
"""
import numpy as np
import pylab
from scipy.optimize import fmin, fmin_ncg, fmin_cg
from scipy import linalg
from scipy import special
from sys import stdout 
from kernels import kernels
from scg import SCG


class GP:
    def __init__(self,X,Y,kernel=None,set_type=1):
        """
            Gaussian Process
        """
        self.N,self.Ydim=Y.shape
        self.setX(X,set_type)
        self.setY(Y,set_type)
        
        if kernel==None:
            self.kernel=kernels(-1,-np.ones(self.Xdim))
            #self.kernel=kernels(-1,-np.ones(self.Xdim),-2.3)
        else: 
            self.kernel=kernel
            
        self.parameter_prior_widths = np.ones(self.kernel.nparams+1)
        self.beta=0.1
        self.update()
        self.n2ln2pi = 0.5*self.Ydim*self.N*np.log(2*np.pi)
        
    def setX(self,newX,set_type=1):
        """"normalization"""
        self.X=newX.copy()
        N,self.Xdim=newX.shape
        self.xmean=self.X.mean(0)
        if set_type==1:
            self.xstd=self.X.std(0)
            self.X-=self.xmean
            self.X/=self.xstd
    
    def setY(self,newY,set_type=1):
        self.Y=newY.copy()
        N,self.Ydim=newY.shape
        self.ymean=self.Y.mean(0)
        if set_type==1:
            self.ystd=self.Y.std(0)
            self.Y-=self.ymean
            self.Y/=self.ystd
    
    def hyper_prior(self):
        #print self.get_params()
        #print self.parameter_prior_widths
        #return -np.dot(self.parameter_prior_widths,np.square(self.get_params()))
        #print -0.5*np.dot(self.parameter_prior_widths,np.square(self.get_params()))
        return -0.5*np.dot(self.parameter_prior_widths,np.square(self.get_params()))
    
    def hyper_prior_grad(self):
        """return the gradient of the (log of the) hyper prior for the current parameters"""
        return -self.parameter_prior_widths*self.get_params()
        #return -self.parameter_prior_widths*1
        
    def get_params(self):
        return np.hstack((self.kernel.get_params(),np.log(self.beta)))
        
    def set_params(self,params):
        assert params.size==self.kernel.nparams+1
        self.beta = np.exp(params[-1])
        self.kernel.set_params(params[:-1])
        
        
    def ll(self,params=None):
        """
         A cost function to optimise for setting the kernel parameters
        """
        if not params == None:
            self.set_params(params)
        try:
            self.update()
        except:
            return np.inf
        
        return -self.marginal()-self.hyper_prior()
        
    def ll_grad(self,params=None):
        """
           the the gradient of the ll function
        """ 
        if not params == None:
            self.set_params(params)
        try:
            self.update()
        except:
            return np.ones(params.shape)*np.NaN
        
        self.update_grad()
        matrix_grads=[e for e in self.kernel.gradients(self.X)]
        matrix_grads.append(-np.eye(self.K.shape[0])/self.beta) #noise gradient matrix
        
        grads=[0.5*np.trace(np.dot(self.alphalphK,e)) for e in matrix_grads]
        
        return -np.array(grads) - self.hyper_prior_grad()
        
        
    def find_kernel_params(self,iters=1000):
        """Optimise the marginal likelihood. work with the log of beta - fmin works better that way.  """
        new_params,final_ll=SCG(self.ll,self.ll_grad,np.hstack((self.kernel.get_params(), np.log(self.beta))),maxiters=iters,display=True,func_flg=1)
        #gtol=1e-10,epsilon=1e-10,
#        new_params = fmin_cg(self.ll,np.hstack((self.kernel.get_params(), np.log(self.beta))),fprime=self.ll_grad,maxiter=iters,gtol=1e-10,disp=False)        
        final_ll=self.ll(new_params)
        return new_params,final_ll
        #print self.kernel.get_params
        
    def update(self):
        """do the Cholesky decomposition as required to make predictions and calculate the marginal likelihood"""
        self.K=self.kernel(self.X,self.X)
        self.K += np.eye(self.K.shape[0])/self.beta
        self.L = np.linalg.cholesky(self.K)
        self.A = linalg.cho_solve((self.L,1),self.Y)
        
    def update_grad(self):
        """do the matrix manipulation required in order to calculate gradients"""
        self.Kinv= np.linalg.solve(self.L.T,np.linalg.solve(self.L,np.eye(self.L.shape[0])))
        self.alphalphK = np.dot(self.A,self.A.T)-self.Ydim*self.Kinv
    
    def marginal(self):
        """The Marginal Likelihood. Useful for optimising Kernel parameters"""
        self.marginal_value=-self.Ydim*np.sum(np.log(np.diag(self.L))) - 0.5*np.trace(np.dot(self.Y.T,self.A)) - self.n2ln2pi
        return self.marginal_value
        
    
        
        
    
    def predict(self,x_star):
        self.update_grad()
        #Make a prediction upon new data points
        #x_star = (np.asarray(x_star)-self.xmean)/self.xstd

        #Kernel matrix k(X_*,X)
        k_x_star_x = self.kernel(x_star,self.X)
        k_x_star_x_star = self.kernel(x_star,x_star)+1./self.beta
        
        #f=self.K*(-0.5*np.log(2*np.pi)-np.square(x_star)/2)
        
        
        #find the means and covs of the projection...
#        means = np.dot(np.dot(k_x_star_x, self.Kinv), self.Y)
#        #means = np.dot(k_x_star_x, self.A)
#        means *= self.ystd
#        means += self.ymean
        
        means=np.hstack(np.dot(np.dot(self.Y.T,self.Kinv),k_x_star_x.T))
        #means *= self.ystd        
        means+=self.ymean
        
        
#        v = np.linalg.solve(self.L,k_x_star_x.T)
#        variances = (np.diag( k_x_star_x_star - np.dot(v.T,v)) + 1./self.beta) * self.ystd
        covs = np.diag( k_x_star_x_star - np.dot(np.dot(k_x_star_x,self.Kinv),k_x_star_x.T))
        return means,covs
        
        
        
#    def predict(self,x_star):
#		"""Make a prediction upon new data points"""
#		x_star = (np.asarray(x_star)-self.xmean)/self.xstd
#
#		#Kernel matrix k(X_*,X)
#		k_x_star_x = self.kernel(x_star,self.X) 
#		k_x_star_x_star = self.kernel(x_star,x_star) 
#		
#		#find the means and covs of the projection...
#		#means = np.dot(np.dot(k_x_star_x, self.K_inv), self.Y)
#		means = np.dot(k_x_star_x, self.A)
#		means *= self.ystd
#		means += self.ymean
#		
#		v = np.linalg.solve(self.L,k_x_star_x.T)
#		#covs = np.diag( k_x_star_x_star - np.dot(np.dot(k_x_star_x,self.K_inv),k_x_star_x.T)).reshape(x_star.shape[0],1) + self.beta
#		variances = (np.diag( k_x_star_x_star - np.dot(v.T,v)).reshape(x_star.shape[0],1) + 1./self.beta) * self.ystd.reshape(1,self.Ydim)
#		return means,variances
  

 
#if __name__=='__main__':
#    Ndata = 50
#    X = np.linspace(-3,3,Ndata).reshape(Ndata,1)
#    Y = np.sin(X) + np.random.standard_normal(X.shape)/20
#    
#    myGP = GP(X,Y)
#    
#    xx = np.linspace(-4,4,200).reshape(200,1)
#    
#    def plot():
#        pylab.plot(X,Y,'r.')
#        yy,cc = myGP.predict(xx)
#        pylab.plot(xx,yy,scaley=False)
#        pylab.plot(xx,yy + 2*np.sqrt(cc),'k--',scaley=False)
#        pylab.plot(xx,yy - 2*np.sqrt(cc),'k--',scaley=False)
#        
#        
#    #plot()
#    #pylab.show()
#    myGP.find_kernel_params()
#    plot()
    
    
    #pylab.show()
    
    
    
    