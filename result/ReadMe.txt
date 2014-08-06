# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 16:38:44 2014
The description of the files
@author: mountain
"""


"""
item_ftr_aft_lsi.txt

format:
[[ftr of the item after lsi] for item in texts]
[[ftr1:1*200],[tr2:1*200]...for i in range(11463)]

load:
corpus_lsi = corpora.MmCorpus(file_path)

#eg:
from gensim import corpora
doc_wordid_path='./doc_ftr_aft_lsi.txt'
corpus_lsi = list(corpora.MmCorpus(doc_wordid_path))

"""

"""
dict.txt
dave the dictionary model obtained by gensim
it contains the information of the words in the document
and the word id of each word

load:
dictionary=corpora.Dictionary.load_from_text(fname)

the usage:
item_word_use_information=dictionary.doc2bow(string)
when given an item description string, we can use the dictionary to convert the string to the
format like
corpus=[(word_id1,occured times1),..]
"""

"""
tfidf_model.txt

load
models.TfidfModel.load(filepath)

usage:
convert a corpus [(word1,occured times),..] to a corpus_tfidf [(word1,tfidf_value),..]
tfidf=models.TfidfModel.load(filepath)
doc_bow = [(0, 1), (1, 1)]  #an item described in the format corpus:[(word1,occured times),..]
corpus_tfidf=tfidf[doc_bow]
print(corpus_tfidf) 
[(0, 0.70710678), (1, 0.70710678)]
"""


"""
item_tfidf.txt

load:
corpus_tfidf=list(corpora.MmCorpus(file_path))

usage:
list(corpora.MmCorpus(doc_wordid_path))
return a list[[(word1,word1_tfidf_value),..] for item in texts]
"""

"""
item_word.json
[[word1,word2...for word in item] for item in texts]
"""

"""
item_wordid.txt
similar to the item_tfidf.txt

load:
corpus=list(corpora.MmCorpus(file_path))
return a list[[(word1,word1_occured_times),..for word1 in item] for item in texts]
"""

"""
lsimodel.txt

load:
lsi = models.LsiModel.load('/tmp/model.lsi')
"""