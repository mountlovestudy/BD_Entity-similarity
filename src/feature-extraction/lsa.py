# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:48:26 2014
Do the LSA using the gensim toolkit
save the LSA information
@author: mountain
"""

from gensim import corpora, models
import json

def LSA(texts):    
    #texts: obtained by data_processing 
    #texts=[[text1],[text2]...[textn] where text1=[word for word occured in text1]
    dictionary=corpora.Dictionary(texts)    
    corpus=[dictionary.doc2bow(text) for text in texts]    
       
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics = 50)
    
    corpus_lsi = lsi[corpus_tfidf]
    
    return dictionary,corpus,tfidf,corpus_tfidf,lsi,corpus_lsi
    
if __name__=='__main__':
    
    #LSA
    f=file(r'../../result/item_word.json')
    texts=json.load(f,encoding = "GB18030", strict=False)
    dictionary,corpus,tfidf,corpus_tfidf,lsi,corpus_lsi=LSA(texts)
    
    #save the dictionary according to the texts
    fname='../../result/dict.txt'
    dictionary.save_as_text(fname)
    
    #save the corpus, corpus=[[(word,word_occured_times_in_text) for word occured in text] for text in texts]
    doc_wordid_path='../../result/item_wordid.txt'
    corpora.MmCorpus.serialize(doc_wordid_path, corpus)    
    #aa=list(corpora.MmCorpus(doc_wordid_path))
    
    #save the tfidf_model which can be used later for mapping a corpus to a tf-idf_copus
    tfidf_model='../../result/tfidf_model.txt'
    tfidf.save(tfidf_model)
    
    #save the corpus_tfidf, list(corpus_tfidf)=[[(word,tf_idf) for word occured in text] for text in texts]
    doc_tfidf_path='../../result/item_tfidf.txt'
    corpora.MmCorpus.serialize(doc_tfidf_path, corpus_tfidf)
    #bb=list(corpora.MmCorpus(doc_tfidf_path))
    
    #save the lsi_model
    lsi_model='../../result/lsimodel.txt'
    lsi.save(lsi_model)
    #lsi = models.LsiModel.load('/tmp/model.lsi')
    
    #save the corpus_lsi,list(corpus_lsi)=[[ftr of the text after lsi] for text in texts]
    corpora.MmCorpus.serialize('../../result/item_ftr_aft_lsi.txt', corpus_lsi)
