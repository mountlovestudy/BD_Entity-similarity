# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 15:15:09 2014
test sklearn.mixture.GMM
@author: mountain
"""

import numpy as np
from sklearn import mixture

np.random.seed(1)
g = mixture.DPGMM(n_components=3)
obs = np.concatenate((np.random.randn(500, 1),-100 + np.random.randn(500, 1),100 + np.random.randn(500, 1)))
g.fit(obs)
print g.n_components
print g.means_
print g.precs_
print g.predict([[0]])
print g.predict_proba([[0]])
