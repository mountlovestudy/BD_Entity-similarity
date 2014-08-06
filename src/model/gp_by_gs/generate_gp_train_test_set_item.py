#!/usr/bin/python
"""
Generate the train set of gp based on item.
The format of training and testing data file is:

# 6000
-- gp_train_label.txt           # 1 0 0 0(1); 0 1 0 0(2); 0 0 0 1(4)
-- gp_train_set.txt             # feature of item1, feature of item2
-- gp_train_set_rot.txt         # feature of item2, feature of item1

# 2000
-- gp_test_label.txt            # 1 0 0 0(1); 0 1 0 0(2); 0 0 0 1(4)
-- gp_test_set.txt              # feature of item1, feature of item2
-- gp_test_set_rot.txt          # feature of item2, feature of item1

All the information needed was already wrote into 
item_ftr_aft_lsi.txt

train item pair set file is in data/train.txt
"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import gensim
import json
import pickle

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

output_train_label_path = '../../../result/gp_train_label.txt'
output_train_label = open(output_train_label_path, 'w')

output_train_set_path = '../../../result/gp_train_set.txt'
output_train_set = file(output_train_set_path, 'w')

output_train_set_rot_path = '../../../result/gp_train_set_rot.txt'
output_train_set_rot = file(output_train_set_rot_path, 'w')

output_test_label_path = '../../../result/gp_test_label.txt'
output_test_label = open(output_test_label_path, 'w')

output_test_set_path = '../../../result/gp_test_set.txt'
output_test_set = file(output_test_set_path, 'w')

output_test_set_rot_path = '../../../result/gp_test_set_rot.txt'
output_test_set_rot = file(output_test_set_rot_path, 'w')

def generate_gp_train_test_set(path_src, path_label, path_set, path_set_rot):
    # generate gp train and test file
    line = path_src.readline()
    line_label = []
    line_set = []
    line_set_rot = []
    while line:
        [item1, item2, label] = line.split(' \t') # label = '1\n' label = label[0]

        line_set_1 = []
        line_set_2 = []

        # write label
        line_label.append(int(label[0]))
                    
        # write set
        feature_of_item1 = corpus_lsi[item2num[item1]]
        feature_of_item2 = corpus_lsi[item2num[item2]]
        for feature_value in feature_of_item1:
            line_set_1.append(feature_value[1])
        for feature_value in feature_of_item2:
            line_set_2.append(feature_value[1])

        line_set.append(line_set_1 + line_set_2)
        line_set_rot.append(line_set_2 + line_set_1)

        line = path_src.readline()

    pickle.dump(line_label, path_label)
    pickle.dump(line_set, path_set)
    pickle.dump(line_set_rot, path_set_rot)
    
    path_src.close()
    path_label.close()
    path_set.close()
    path_set_rot.close()


if __name__ == '__main__':

    generate_gp_train_test_set(train_src, output_train_label, output_train_set, output_train_set_rot)

    generate_gp_train_test_set(test_src, output_test_label, output_test_set, output_test_set_rot)

