#!/usr/bin/python
"""
File result/word/word_ftr_aft_lsi.txt contains the word feature:
70000 x 200

then use kmeans to get K centers.
"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import gensim
import json
import numpy
from read_and_write import write_dict_into_GB18030

if __name__ == '__main__':
    '''
    '''
    # word_feature_path = '../../../result/word/word_ftr_aft_lsi.txt'
    # corpus_lsi = list(gensim.corpora.MmCorpus(word_feature_path))


    f=file(r'../../result/pre/words_id_dict_further.json')
    #f=file(r'E:/BD/bddac/result/pre/test.json')
    dictword=json.load(f,encoding = "GB18030", strict=False)

    # item2num_path = file('../../../result/item2num.json')
    # item2num = json.load(item2num_path) # item2num['item'] = num

    # output_path = '../../../result/svm_test_set_item.txt'
    # output = file(output_path2, 'w')
    
    word_num={}
    for i,k in enumerate(dictword):
        word_num[k]=i
     
    word2num_path = '../../result/word2num.json'
    write_dict_into_GB18030(word2num_path, word_num)
