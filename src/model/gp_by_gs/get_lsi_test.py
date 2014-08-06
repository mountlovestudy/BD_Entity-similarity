# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:58:39 2014
get test
@author: mountain
"""
import os.path
import pickle
from gensim import corpora
import numpy as np

import json


def get_lsi_test(test_file,item2num_file,lsi_file):
    #return lsi_ftr for train
    assert os.path.exists(test_file), "test file doesn't exist"
    assert os.path.exists(item2num_file), "item2num_file doesn't exist"
    assert os.path.exists(lsi_file), "lsi_file doesn't exist"
    info=open(test_file,"r")
    item2num=json.load(file(item2num_file))
    corpus_lsi = list(corpora.MmCorpus(lsi_file))
    ftr=[]
    for line in info:
        id_lbl=line.split()
        ftrnow=[k[1] for k in corpus_lsi[item2num[id_lbl[0]]]]+[g[1] for g in corpus_lsi[item2num[id_lbl[1]]]]       
        ftr.append(ftrnow)
    info.close()
    ftr=np.array(ftr)
    return ftr
    
if __name__=="__main__":
    test_file="../../../data/train.txt"
    lsi_file="../../../result/item_ftr_aft_lsi.txt"
    item2num_file="../../../result/item2num.json"
    ftr=get_lsi_test(test_file,item2num_file,lsi_file)
    test_save="../../../src/model/gp/test_fnl.txt"
    output=open(test_save,'w')
    pickle.dump(ftr, output)
    output.close()

    
        
    
