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

if __name__ == '__main__':
    '''
    '''
    item_feature_path = '../../../result/item_ftr_aft_lsi.txt'
    corpus_lsi = list(gensim.corpora.MmCorpus(item_feature_path))

    train_src_path = '../../../data/svm_train_6000.txt'
    train_src = open(train_src_path, 'r')

    test_src_path = '../../../data/svm_test_2000.txt'
    test_src = open(test_src_path, 'r')

    item2num_path = file('../../../result/item2num.json')
    item2num = json.load(item2num_path) # item2num['item'] = num
    
    output_path1 = '../../../result/svm_train_set_item.txt'
    output1 = file(output_path1, 'w')

    output_path2 = '../../../result/svm_test_set_item.txt'
    output2 = file(output_path2, 'w')

    # generate svm train file
    line = train_src.readline()
    while line:
        [item1, item2, label] = line.split(' \t') # label = '1\n' label = label[0]

        line_output = ''
        line_output = line_output + label[0] + ' '
        # use 'label 1:x1 ... 400:x400'
        feature_of_item1 = corpus_lsi[item2num[item1]]
        feature_of_item2 = corpus_lsi[item2num[item2]]
        # for feature_value in feature_of_item1:
        #     line_output = line_output + str(feature_value[0] + 1) + ':' + str(feature_value[1]) + ' '
        # for feature_value in feature_of_item2:
        #     line_output = line_output + str(feature_value[0] + 201) + ':' + str(feature_value[1]) + ' '
        
        # use feature1 - feature2 as new feature
        if len(feature_of_item1) == len(feature_of_item2):
            for i in range(len(feature_of_item1)):
                line_output = line_output + str(i + 1) + ':' + str(abs(feature_of_item1[i][1] - feature_of_item2[i][1])) + ' '

        line_output = line_output + '\n'    
        output1.writelines(line_output)
        line = train_src.readline()

    train_src.close()
    output1.close()
    
    # generate svm test file
    line = test_src.readline()
    while line:
        [item1, item2, label] = line.split(' \t') # label = '1\n' label = label[0]

        line_output = ''
        line_output = line_output + label[0] + ' '
        feature_of_item1 = corpus_lsi[item2num[item1]]
        feature_of_item2 = corpus_lsi[item2num[item2]]

        # for feature_value in feature_of_item1:
        #     line_output = line_output + str(feature_value[0] + 1) + ':' + str(feature_value[1]) + ' '
        # for feature_value in feature_of_item2:
        #     line_output = line_output + str(feature_value[0] + 201) + ':' + str(feature_value[1]) + ' '

        # use feature1 - feature2 as new feature
        if len(feature_of_item1) == len(feature_of_item2):
            for i in range(len(feature_of_item1)):
                line_output = line_output + str(i + 1) + ':' + str(abs(feature_of_item1[i][1] - feature_of_item2[i][1])) + ' '

        line_output = line_output + '\n'    
        output2.writelines(line_output)
        line = test_src.readline()

    test_src.close()
    output2.close()    

