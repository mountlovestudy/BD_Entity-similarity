#!/usr/bin/python
"""
Generate the train set of svm based on item.
The format of training and testing data file is:

<label-1> <index1>:<value1> <index2>:<value2> ...
.
.
<label-2> <index1>:<value1> <index2>:<value2> ...
.
.

value is a real number of the feature, which was 
already extracted by lsa and wrote into 
item_ftr_aft_lsi.txt

train item pair set file is in data/train.txt
"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import gensim
import json
import numpy
import jieba
import codecs
import linecache
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

train_src_path = '../../../data/svm_train_6000.txt'
# train_src = open(train_src_path, 'r')

test_src_path = '../../../data/svm_test_2000.txt'
# test_src = open(test_src_path, 'r')

word2num_path = file('../../../result/word2num.json')
word2num = json.load(word2num_path, encoding = "GB18030", strict=False) # item2num['item'] = num

output_path1 = '../../../result/svm_train_set_word.txt'
# output1 = file(output_path1, 'w')

output_path2 = '../../../result/svm_test_set_word.txt'
# output2 = file(output_path2, 'w')

entity_file_path = '../../../data/entity.json'
# entity_file = open(entity_file_path)
# stop_words_path = '../../../data/stopwords.txt'

word_feature_path = '../../../result/word/word_ftr_aft_lsi.txt'
corpus_lsi = list(gensim.corpora.MmCorpus(word_feature_path))

def get_words(entity_file_path, item1, item2):
    entity_file = open(entity_file_path)
    words1 = []
    words2 = []
    for line in entity_file.readlines():
        entity = line.split(' \t')
        entity_id = entity[0]
        if entity_id == item1:
            entity_json = json.loads(entity[1].decode("GB18030"), encoding = "GB18030", strict=False)
            for value in entity_json.values():
                for item in value:
                    seg_list = jieba.cut(item.encode("GB18030"))
                    for word in seg_list:
                        words1.append(word)

        if entity_id == item2:
            entity_json = json.loads(entity[1].decode("GB18030"), encoding = "GB18030", strict=False)
            for value in entity_json.values():
                for item in value:
                    seg_list = jieba.cut(item.encode("GB18030"))
                    for word in seg_list:
                        words2.append(word)
    return words1, words2

def generate_svm_set(src_path, output_path):
    src = open(src_path, 'r')
    output = file(output_path, 'w')
    line = src.readline()
    while line:
        [item1, item2, label] = line.split(' \t') # label = '1\n' label = label[0]

        line_output = ''
        line_output = line_output + label[0] + ' '
        # use 'label 1:x1 ... 400:x400'
        words1, words2 = get_words(entity_file_path, item1, item2)

        # generate item1's feature, all words in item1 +
        cnt1 = 0
        feature_of_word1 = numpy.matrix(numpy.zeros((1,200)))
        for word in words1:
            if word in word2num.keys():
                if len(corpus_lsi[word2num[word]]) == 200:
                    feature_of_word1 = feature_of_word1 + numpy.matrix(numpy.array(corpus_lsi[word2num[word]])[:,1])
                    cnt1 = cnt1 + 1
        if not cnt1 == 0:
            feature_of_word1 = feature_of_word1/cnt1

        cnt2 = 0
        feature_of_word2 = numpy.matrix(numpy.zeros((1,200)))
        for word in words2:
            if word in word2num.keys():
                if len(corpus_lsi[word2num[word]]) == 200:
                    feature_of_word2 = feature_of_word2 + numpy.matrix(numpy.array(corpus_lsi[word2num[word]])[:,1])
                    cnt2 = cnt2 + 1
        if not cnt2 == 0:
            feature_of_word2 = feature_of_word2/cnt2
        
        # for feature_value in feature_of_item1:
        #     line_output = line_output + str(feature_value[0] + 1) + ':' + str(feature_value[1]) + ' '
        # for feature_value in feature_of_item2:
        #     line_output = line_output + str(feature_value[0] + 201) + ':' + str(feature_value[1]) + ' '
        
        # use feature1 - feature2 as new feature
        diff = abs(feature_of_word1 - feature_of_word2)
        if not 0 == numpy.max(diff):
            diff = (diff - numpy.mean(diff)) / numpy.max(diff)
        for i in range(diff.shape[1]):
            line_output = line_output + str(i + 1) + ':' + str(diff[0, i]) + ' '

        line_output = line_output + '\n'
        output.writelines(line_output)
        line = src.readline()

    src.close()
    output.close()

if __name__ == '__main__':
    '''
    '''

    # generate svm train file
    generate_svm_set(train_src_path, output_path1)

    # generate test svm set
    generate_svm_set(test_src_path, output_path2)
