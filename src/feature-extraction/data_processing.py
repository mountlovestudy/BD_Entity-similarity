# -*- coding: utf-8 -*-
"""
Created on Wed Jul 09 20:51:46 2014
Pre-processing the json data
get a list texts=[[text1],[text2]...[textn] where text1=[word for word occured in text1]
get a dict id2num={id:num}, it's a id-num-mapping
@author: mountain
"""
import json
import jieba
from read_and_write import write_dict_into_GB18030, get_stop_words, write_dict_into_utf8


def get_text_in_line(line,stopwords):
        entity = line.split(' \t')
        entity_id = entity[0]
        entity_json = json.loads(entity[1].decode("GB18030"), encoding = "GB18030", strict=False)
        # entity_json = json.loads(entity[1])
        
        text=[]
        for value in entity_json.values():
            for item in value:
                #segs=jieba.cut(item.encode("GB18030"))
                #for seg in segs:
                 #   print seg
                 #   if seg not in stopwords:
                  #      tmp.append(seg)
                tmp=[seg for seg in jieba.cut(item.encode("GB18030")) if seg+'\n' not in stopwords]
            text.append(tmp)
        text=sum(text,[])
        return entity_id,text
            



def get_dict(doc_path,stop_word_path):
    #stop_word_path='../data/stopwords.txt'
    stopwords=set(get_stop_words(stop_word_path))
    
    doc=open(doc_path)
    cnt=0
    texts=[]
    
    num2id={}
    id2num={}
    
    for line in doc.readlines():
        entity_id,line_text=get_text_in_line(line,stopwords)
        texts.append(line_text)
        num2id[cnt]=entity_id
        id2num[entity_id]=cnt
        cnt=cnt+1
        
    
    all_word=sum(texts,[])
    word_once=set(word for word in set(all_word) if all_word.count(word)==1)
    texts=[[word for word in text if word not in word_once] 
            for text in texts]
    
    return texts,id2num

if __name__=='__main__':
    #test
    #doc_path = '../data/entity-test.json'
    doc_path = '../data/entity.json'
    stop_word_path='../data/stopwords.txt'
    texts,id2num=get_dict(doc_path,stop_word_path)
    doc_word_path='../result/doc_word.json'
    write_dict_into_GB18030(doc_word_path,texts)
    id2num_path='../result/id2num.json'
    write_dict_into_utf8(id2num_path,id2num)

    
    
    
    
