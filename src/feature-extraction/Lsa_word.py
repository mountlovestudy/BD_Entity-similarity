# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:19:02 2014
Do the LSA using the gensim toolkit
save the LSA information
the texts is built in form of:
    [[the id which contains the word1],[the id which contains the word2]...]
this versions is for words analysis
@author: mountain
"""
from gensim import corpora
from LSA import LSA
import json
from read_and_write import write_dict_into_GB18030



f=file(r'../../result/pre/words_id_dict_further.json')
#f=file(r'E:/BD/bddac/result/pre/test.json')
dictword=json.load(f,encoding = "GB18030", strict=False)

f=file(r'../../result/pre/id_num_mapping.json')
idnum=json.load(f,encoding = "GB18030", strict=False)

word_num={}
texts=[]
for i,k in enumerate(dictword):
    word_num[k]=i
    tmp=[]
    for s in idnum:
        for ele in dictword[k]:
            if idnum[s]==ele:
                tmp.append(s)
    texts.append(tmp)

word_doc_path='../../result/word/word_doc.json'
write_dict_into_GB18030(word_doc_path,texts) 

word2num_path = '../../result/word2num.json'
write_dict_into_GB18030(word2num_path, word_num)

dictionary,corpus,tfidf,corpus_tfidf,lsi,corpus_lsi=LSA(texts)

#save the dictionary according to the texts
fname='../../result/word/dict.txt'
dictionary.save_as_text(fname)

#save the corpus, corpus=[[(id,id_occured_times_in_word) for id occured in word] for word in texts]
word_docid_path='../../result/word/word_docid.txt'
corpora.MmCorpus.serialize(word_docid_path, corpus)    
#aa=list(corpora.MmCorpus(word_docid_path))

#save the tfidf_model which can be used later for mapping a corpus to a tf-idf_copus
tfidf_model='../../result/word/tfidf_model.txt'
tfidf.save(tfidf_model)

#save the corpus_tfidf, list(corpus_tfidf)=[[(id,tf_idf) for id occured in word] for word in texts]
word_tfidf_path='../../result/word/doc_tfidf.txt'
corpora.MmCorpus.serialize(word_tfidf_path, corpus_tfidf)
#bb=list(corpora.MmCorpus(word_tfidf_path))

#save the lsi_model
lsi_model='../../result/word/lsimodel.txt'
lsi.save(lsi_model)
#lsi = models.LsiModel.load('/tmp/model.lsi')

#save the corpus_lsi,list(corpus_lsi)=[[ftr of the word after lsi] for word in texts]
corpora.MmCorpus.serialize('../../result/word/word_ftr_aft_lsi.txt', corpus_lsi)
    
