# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 12:01:37 2014

@author: mountain
"""
import os.path
import pickle
from gensim import corpora
import numpy as np

import json


def get_lsi_train_wth_differ_lbl(train_file,item2num_file,lsi_file):
    #return lsi_ftr for train
    assert os.path.exists(train_file), "train file doesn't exist"
    assert os.path.exists(item2num_file), "item2num_file doesn't exist"
    assert os.path.exists(lsi_file), "lsi_file doesn't exist"
    info=open(train_file,"r")
    item2num=json.load(file(item2num_file))
    corpus_lsi = list(corpora.MmCorpus(lsi_file))
    ftr={1:[],2:[],3:[],4:[]}
    for line in info:
        id_lbl=line.split()
        ftrnow=[k[1] for k in corpus_lsi[item2num[id_lbl[0]]]]+[g[1] for g in corpus_lsi[item2num[id_lbl[1]]]]       
        clsnow=id_lbl[2]
        if clsnow=='1':
            ftr[1].append(ftrnow)
        elif clsnow=='2':
            ftr[2].append(ftrnow)
        elif clsnow=='3':
            ftr[3].append(ftrnow)
        else:
            ftr[4].append(ftrnow)
    info.close()
    return ftr
    
if __name__=="__main__":
    train_file="../../../data/train.txt"
    lsi_file="../../../result/item_ftr_aft_lsi.txt"
    item2num_file="../../../result/item2num.json"
    ftr=get_lsi_train_wth_differ_lbl(train_file,item2num_file,lsi_file)
    train_save="../../../src/model/gp/train_ftr_differ_lbl.txt"
    output=open(train_save,'w')
    pickle.dump(ftr, output)
    output.close()
#    output=open(cls_save,'w')
#    pickle.dump(cls, output)
#    output.close()
    
        
    
