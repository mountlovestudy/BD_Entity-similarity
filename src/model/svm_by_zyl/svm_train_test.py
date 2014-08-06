#!/usr/bin/python
"""
Use multi class svm to train and test our data.
refer to http://svmlight.joachims.org/svm_multiclass.html

"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import numpy
import sys, os.path

svmtrain_exe = './svm_multiclass_learn'
svmpredict_exe = './svm_multiclass_classify'

# train_set = '../../../result/svm_train_set_item.txt'
# test_set = '../../../result/svm_test_set_item.txt'
# svm_model = '../../../result/svm_train_item.model'

train_set = '../../../result/svm_train_set_word.txt'
test_set = '../../../result/svm_test_set_word.txt'
svm_model = '../../../result/svm_train_word.model'

pass_through_options = '-c 10 -t 2'

output = 'out'
if __name__ == '__main__':
    '''
    '''
    print "Training problem for label %s..."
    cmd = "%s %s %s %s" % (svmtrain_exe, pass_through_options, train_set, svm_model)
    os.system(cmd)


    print "Testing problem for label %s..."
    cmd = "%s %s %s %s" % (svmpredict_exe, test_set, svm_model, output)
    os.system(cmd)
