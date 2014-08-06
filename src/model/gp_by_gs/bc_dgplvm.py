# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 18:15:11 2014

@author: mountain
"""



import numpy as np
import pylab
from PCA_EM import PCA_EM
import GP
from scg import SCG
from scipy import optimize,special
import MLP
from GPc import GPc
from sklearn.decomposition import PCA





class GPLVM:
    def __init__(self,Y,dim):
        self.Xdim=dim
        self.N,self.Ydim=Y.shape
        
        
        
        """Use PCA to initalise the problem. Uses EM version in this case..."""
        myPCA_EM = PCA_EM(Y,dim)
        myPCA_EM.learn(100) 
        X = np.array(myPCA_EM.m_Z)
        
        self.GP = GP.GP(X,Y)
        
    def learn(self,niters):
        for i in range(niters):
            self.optimise_latents()
            self.optimise_GP_kernel()
            
    def optimise_GP_kernel(self):
        self.GP.find_kernel_params()
        print self.GP.marginal(), 0.5*np.sum(np.square(self.GP.X))
        
    def ll(self,xx,i):
        """The log likelihood function - used when changing the ith latent variable to xx"""
        self.GP.X[i] = xx
        self.GP.update()
        return -self.GP.marginal()+ 0.5*np.sum(np.square(xx))
        
    def ll_grad(self,xx,i):
        """the gradient of the likelihood function for us in optimisation"""
        self.GP.X[i]=xx
        self.GP.update()
        self.GP.update_grad()
        matrix_grads = [self.GP.kernel.gradients_wrt_data(self.GP.X,i,jj) for jj in range(self.GP.Xdim)]
        grads = [-0.5*np.trace(np.dot(self.GP.alphalphK,e)) for e in matrix_grads]
        return np.array(grads)+xx
        
        
    def optimise_latents(self):
        """Direct optimisation of the latents variables."""
        xtemp=np.zeros(self.GP.X.shape)
        for i,yy in enumerate(self.GP.Y):
            original_x = self.GP.X[i].copy()
            #gtol=1e-10,epsilon=1e-10,
            xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,disp=True,args=(i,))
            #xopt=SCG(self.ll,self.ll_grad,self.GP.X[i],optargs=(i,),display=False)
            self.GP.X[i] = original_x
            xtemp[i] = xopt
            
        self.GP.X=xtemp.copy()

class DGPLVM(GPLVM):
    """A(back) constrained version of the GPLVM"""
    def __init__(self,data,xdim,cls,delta,nhidden=5,mlp_alpha=2):
        GPLVM.__init__(self,data,xdim)
		
        self.cls=cls
        #self.cls_num={1:0,2:0,3:0,4:0}
        self.cls_num={1:0,2:0}
        #the para in P(X)
        self.delta=delta 
        
        for i in cls:
            self.cls_num[i]=self.cls_num[i]+1
            
        self.MLP = MLP.MLP((self.Ydim,nhidden,self.Xdim),alpha=mlp_alpha)
        self.MLP.train(self.GP.Y,self.GP.X)#create an MLP initialised to the PCA solution...
        self.GP.X = self.MLP.forward(self.GP.Y)
		
	def unpack(self,w):
		""" Unpack the np array into the free variables of the current instance"""
		assert w.size == self.MLP.nweights + self.GP.kernel.nparams + 1,"bad number of parameters for unpacking"
		self.MLP.unpack(w[:self.MLP.nweights])
		self.GP.X = self.MLP.forward(self.GP.Y)
		self.GP.set_params(w[self.MLP.nweights:])
		
	def pack(self):
		""" 'Pack up' all of the free variables in the model into a np array"""
		return np.hstack((self.MLP.pack(),self.GP.get_params()))
	



    def meanX(self):
        self.meanall=self.GP.X.mean(0)
        
        
    def meanxi(self,i):
        #the mean of the ith class
        
        return np.dot(self.cls==i,self.GP.X)/self.cls_num[i]
        
    def SwSb(self):
        
        Sb=np.sum(np.square(self.mean_cls[1]-self.mean_cls[2]))
        
        w_diff={1:0,2:0}
        for i in range(self.N):
            diff=self.GP.X[i]-self.mean_cls[self.cls[i]]
            w_diff[self.cls[i]]=w_diff[self.cls[i]]+np.dot(diff,diff)
        
        Sw=w_diff[1]+w_diff[2]
        
        self.Sb=Sb
        self.Sw=Sw
                
        
        
    def x_prior(self):
        self.meanX()
        #self.mean_cls={1:self.meanxi(1),2:self.meanxi(2),3:self.meanxi(3),4:self.meanxi(4)}
        self.mean_cls={1:self.meanxi(1),2:self.meanxi(2)}
        self.SwSb()
        self.x_prior_value=-self.Sw/(self.Sb*(self.delta))
        return self.x_prior_value
        
    def x_prior_grad(self):
        #x_l: the class of x
        self.meanX()
        self.mean_cls={1:self.meanxi(1),2:self.meanxi(2)}
        self.SwSb()
        
        x_p_g=np.zeros(self.GP.X.shape)
        for i,xx in enumerate(self.GP.X):
            x_l=self.cls[i]
            sb_grad=np.zeros(len(xx))
            sw_grad=sb_grad
            if x_l==1:
                sb_grad=2*(self.mean_cls[1]-self.mean_cls[2])/self.cls_num[1]
                sw_grad=2*(xx-self.mean_cls[1])
            else:
                sb_grad=-2*(self.mean_cls[1]-self.mean_cls[2])/self.cls_num[2]
                sw_grad=2*(xx-self.mean_cls[2])
            x_prior_grad=-1./(self.delta*np.square(self.Sb))*(sw_grad*self.Sb-self.Sw*sb_grad)
            x_p_g[i]=x_prior_grad
        return x_p_g    
        
	def ll(self,w):
		"""Calculate and return the -ve log likelihood of the model (actually, the log probabiulity of the model). To be used in optimisation routine"""
		self.unpack(w)
		self.GP.update()
		return  self.GP.ll() - self.x_prior -self.MLP.prior()
	
	
	def ll_grad(self,w):
		"""The gradient of the ll function - used for quicker optimisation via fmin_cg"""
		self.unpack(w)
		self.GP.update()
		#gradients wrt the GP parameters can be done inside the GP class. This also updates the GP, computes alphalphK.
		GP_grads = self.GP.ll_grad(w[self.MLP.nweights:])

		#gradient matrices (gradients of the kernel matrix wrt data)
		gradient_matrices = self.GP.kernel.gradients_wrt_data(self.GP.X)
		 
		#gradients of the error function wrt 'network outputs', i.e. latent variables
		x_gradients = np.array([-0.5*np.trace(np.dot(self.GP.alphalphK,e)) for e in gradient_matrices]).reshape(self.GP.X.shape)-self.x_prior_grad()
         
		
		#backpropagate...
		weight_gradients = self.MLP.backpropagate(self.GP.Y,x_gradients) - self.MLP.prior_grad()
		return np.hstack((weight_gradients,GP_grads))
		
	def learn(self,callback=None,gtol=1e-4):
		"""'Learn' by optimising the weights of the MLP and the GP hyper parameters together.  """
		w_opt = optimize.fmin_cg(self.ll,np.hstack((self.MLP.pack(),self.GP.kernel.get_params(),np.log(self.GP.beta))),self.ll_grad,args=(),callback=callback,gtol=gtol)
		final_cost = self.ll(w_opt)#sets all the parameters...
              
        
        
    