# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 16:19:51 2014

The GPLVM refered to the work by James Hensman

Add the DGPLVM model

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
    
    def get_X(self):
        return GP.X


class DGPLVM:
    def __init__(self,Y,dim,cls,delta,nhidden=5,mlp_alpha=2):
        
        #cls is the class of each y in Y, a vector 
        self.Xdim=dim
        self.N,self.Ydim=Y.shape
        self.cls=cls
        #self.cls_num={1:0,2:0,3:0,4:0}
        self.cls_num={1:0,2:0}
        #the para in P(X)
        self.delta=delta 
        
        for i in cls:
            self.cls_num[i]=self.cls_num[i]+1
        
        """Use PCA to initalise the problem. Uses EM version in this case..."""
#        myPCA_EM = PCA_EM(Y,dim)
#        myPCA_EM.learn(100) 
#        X = np.array(myPCA_EM.m_Z)
        
        self.pca=PCA(n_components=dim)
        X=self.pca.fit_transform(Y)
        self.GP = GP.GP(X,Y)
              
        
        
    def learn(self,niters):
        self.f_kernel=[]
        self.f_x=[]
        for i in range(niters):
            print i
            self.optimise_latents()
            self.optimise_GP_kernel()
            if i>5:
                if self.f_kernel[-1]>self.f_kernel[-2] and self.f_kernel[-2]>self.f_kernel[-3] and self.f_x[-1]>self.f_x[-2] and self.f_x[-2]>self.f_x[-3]:
                    break
            
    def optimise_GP_kernel(self):
        para_new,f_new=self.GP.find_kernel_params()
        #print self.GP.marginal(), 0.5*np.sum(np.square(self.GP.X))
        self.f_kernel.append(f_new)
        
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
        
    def x_prior_grad(self,xx,x_l):
        #x_l: the class of x
        self.meanX()
        #self.mean_cls={1:self.meanxi(1),2:self.meanxi(2),3:self.meanxi(3),4:self.meanxi(4)}
        self.mean_cls={1:self.meanxi(1),2:self.meanxi(2)}
        self.SwSb()
        sb_grad=np.zeros(len(xx))
        sw_grad=sb_grad
        if x_l==1:
            sb_grad=2*(self.mean_cls[1]-self.mean_cls[2])/self.cls_num[1]
            sw_grad=2*(xx-self.mean_cls[1])
        else:
            sb_grad=-2*(self.mean_cls[1]-self.mean_cls[2])/self.cls_num[2]
            sw_grad=2*(xx-self.mean_cls[2])
                
        
        x_prior_grad=-1./(self.delta*np.square(self.Sb))*(sw_grad*self.Sb-self.Sw*sb_grad)
        return x_prior_grad
        
    def ll(self,xx,i,xx_l,update=1):
        """The log likelihood function - used when changing the ith latent variable to xx"""
        if update==1:
            self.GP.X[i] = xx
            self.GP.update()
            return -self.GP.marginal()- self.x_prior()
        else:
            return -self.GP.marginal_value-self.x_prior_value
            
    
    def ll_grad(self,xx,i,xx_l,update=1):
        """the gradient of the likelihood function for us in optimisation"""
        if update==1:
            self.GP.X[i]=xx
            self.GP.update()
        
        self.GP.update_grad()
                
        matrix_grads = [self.GP.kernel.gradients_wrt_data(self.GP.X,i,jj) for jj in range(self.GP.Xdim)]
        grads = [-0.5*np.trace(np.dot(self.GP.alphalphK,e)) for e in matrix_grads]
                
        return np.array(grads)-self.x_prior_grad(xx,xx_l)
        
        
    def optimise_latents(self):
        """Direct optimisation of the latents variables."""
        xtemp=np.zeros(self.GP.X.shape)
        for i,yy in enumerate(self.GP.Y):
            original_x = self.GP.X[i].copy()
            xx_l=self.cls[i]
            #xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,gtol=1e-10,disp=True,args=(i,xx_l))
            xopt,fxnew=SCG(self.ll,self.ll_grad,self.GP.X[i],optargs=(i,xx_l),display=False,func_flg=1)
            self.GP.X[i] = original_x
            xtemp[i] = xopt
        
        
        self.GP.X=xtemp.copy()
        self.f_x.append(fxnew)
    
    """
        ynew to xnew, using back constrains
    """
    def MLP_model(self,nhidden,mlp_alpha):
        self.MLP = MLP.MLP((self.Ydim,nhidden,self.Xdim),alpha=mlp_alpha)
        self.MLP.train(self.GP.Y,self.GP.X)#create an MLP initialised to the PCA solution...
    
    def predict(self,ynew,nhidden=5,mlp_alpha=2):
        self.MLP_model(nhidden,mlp_alpha)
        ynew-=self.GP.ymean
        ynew/=self.GP.ystd
        xnew = self.MLP.forward(ynew)
        return xnew
#        
    """
     using optimization
    """
    def predict_ll(self,xnew,ynew):
        u,deta=self.GP.predict(xnew)
        val=np.sum(np.square(ynew-u))/(2*deta)+0.5*self.GP.Ydim*np.log(deta)
        return val
        
    def predict_x(self,ynew):
        #x=np.zeros([len(ynew),self.GP.Xdim])
        ynew=ynew-self.GP.ymean
        ynew=ynew/self.GP.ystd
        x=self.pca.transform(ynew)
        x_latent=np.zeros(x.shape)
        for i,xtmp in enumerate(x):
            xnew=optimize.fmin_bfgs(self.predict_ll,xtmp,args=(ynew[i],))
            x_latent[i]=xnew.copy()
        return x_latent
    
    
    def predict_x_union(self,ynew):
        ynew=ynew-self.GP.ymean
        ynew=ynew/self.GP.ystd
        #xnew=self.pca.fit_transform(ynew)
        xnew=self.GP.X.mean(0)*np.ones([len(ynew),self.GP.Xdim])
        X_new=np.concatenate((self.GP.X,xnew),axis=0)
        Y_new=np.concatenate((self.GP.Y,ynew),axis=0)
        self.GP_new=GP.GP(X_new,Y_new,set_type=1)
        self.GP_new.set_params(self.GP.get_params())
        self.GP_new.update()
        
        for j in range(10):
            xtemp = np.zeros(self.GP_new.X.shape)
            for i in range(self.GP.N,self.GP_new.N):
                original_x = self.GP_new.X[i].copy()
                #xopt = optimize.fmin_cg(self.ll,self.GP.X[i],fprime=self.ll_grad,gtol=1e-10,disp=True,args=(i,xx_l))
                xopt,fxnew=SCG(self.predict_x_ll,self.predict_ll_grad,self.GP_new.X[i],optargs=(i,),display=False,func_flg=1)
                self.GP_new.X[i] = original_x
                xtemp[i] = xopt
            self.GP_new.X=xtemp.copy()
        return self.GP_new.X[self.GP.N:self.GP_new.N,:]
        
        
    def predict_x_ll(self,xx,i):
        """The log likelihood function - used when changing the ith latent variable to xx"""
        self.GP_new.X[i] = xx
        self.GP_new.update()
        return -self.GP_new.marginal()
        
    def predict_ll_grad(self,xx,i):
		"""the gradient of the likelihood function for us in optimisation"""
		self.GP_new.X[i] = xx
		self.GP_new.update()
		self.GP_new.update_grad()
		matrix_grads = [self.GP_new.kernel.gradients_wrt_data(self.GP_new.X,i,jj) for jj in range(self.GP_new.Xdim)]
		grads = [-0.5*np.trace(np.dot(self.GP_new.alphalphK,e)) for e in matrix_grads]
		return np.array(grads)
        
        
        
    def predict_gs(self,ynew):
        self.GP_yx=GP.GP(self.GP.Y,self.GP.X)
        self.GP_yx.find_kernel_params()
        xnew=self.GP_yx.predict(ynew)
        return xnew
        
        

        
    def classify(self,ynew,maxiter=10000):
        self.xnew=self.predict(ynew)    
        self.cls=-(self.cls-1.5)*2
        g=GPc(self.GP.K,self.cls)
        g.cal_f_w(maxiter)
        
##        g=GPc(self.GP.K,self.cls)
##        f,W=g.cal_f_w()
        Kp=self.GP.K+np.linalg.inv(g.W)
        Kinv= np.linalg.solve(self.GP.L.T,np.linalg.solve(self.GP.L,np.eye(self.GP.L.shape[0])))
#        Kl=np.linalg.cholesky(Kp)
#        Kpinv=np.linalg.solve(Kl.T,np.linalg.solve(Kl,np.eye(Kl.shape[0])))
        Kpinv=np.linalg.inv(Kp)
#        
        y=[]
        xx=self.xnew.copy()
        for i in range(len(xx)):
            xnew=xx[i]
            #xnew = (np.asarray(xnew)-self.GP.xmean)/self.GP.xstd
            k_star=self.GP.kernel(xnew,self.GP.X)
            k_star_star=self.GP.kernel(xnew,xnew)+np.eye(1)/self.GP.beta
            means = np.dot(k_star, Kinv)
            means=np.dot(means,g.fnew)
            var=k_star_star-np.dot(np.dot(k_star,Kpinv),k_star.T)
            y.append(special.ndtr(means/np.sqrt(1+var)))        
        return y
        
        
#        y=[]
#        for i in range(len(xnew)):
#            xstar=xnew[i]
##            xstar=(np.asarray(xnew)-self.GP.xmean)/self.GP.xstd
#            k_star=self.GP.kernel(xstar,self.GP.X)
#            k_star_star=self.GP.kernel(xstar,xstar)+1/self.GP.beta
#            fstar=np.dot(k_star,g.delta)
#            v=np.dot(np.linalg.inv(g.L),np.dot(np.sqrt(g.W),k_star.T))
#            Vf=k_star_star-np.dot(v.T,v)
#            y.append(special.ndtr(fstar/np.sqrt(1+Vf))[0])
#        return y
            


#if __name__=="__main__":
#	N = 30
#	colours = np.arange(N)#something to colour the dots with...
#	theta = np.linspace(2,6,N)
#	Y = np.vstack((np.sin(theta)*(1+theta),np.cos(theta)*theta)).T
#	Y += 0.1*np.random.randn(N,2)
#	
#	thetanorm = (theta-theta.mean())/theta.std()
#	
#	xlin = np.linspace(-1,1,1000).reshape(1000,1)
#	
#	myGPLVM = GPLVM(Y,1)
#	
#	def plot_current():
#		pylab.figure()
#		ax = pylab.axes([0.05,0.8,0.9,0.15])
#		pylab.scatter(myGPLVM.GP.X[:,0]/myGPLVM.GP.X.std(),np.zeros(N),40,colours)
#		pylab.scatter(thetanorm,np.ones(N)/2,40,colours)
#		pylab.yticks([]);pylab.ylim(-0.5,1)
#		ax = pylab.axes([0.05,0.05,0.9,0.7])
#		pylab.scatter(Y[:,0],Y[:,1],40,colours)
#		Y_pred = myGPLVM.GP.predict(xlin)[0]
#		pylab.plot(Y_pred[:,0],Y_pred[:,1],'b')
#	
#	class callback:
#		def __init__(self,print_interval):
#			self.counter = 0
#			self.print_interval = print_interval
#		def __call__(self,w):
#			self.counter +=1
#			if not self.counter%self.print_interval:
#				print self.counter, 'iterations, cost: ',myGPLVM.GP.get_params()
#				plot_current()
#				
#	#cb = callback(100)
#			
#	myGPLVM.learn(30)
#	plot_current()
#	
#	pylab.show()
#		


  

		
