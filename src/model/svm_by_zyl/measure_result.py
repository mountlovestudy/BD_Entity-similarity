#!/usr/bin/python
"""
Compute the precision and score.
precision = TP/N                # N is the number of all test sample(2000/1991)
score = D = sqrt(sum(Sc_i-Smi)^2)

"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import numpy

if __name__ == '__main__':
    '''
    '''
    test_src_path = '../../../data/svm_test_2000.txt'
    test_src = open(test_src_path, 'r')

    result_path = 'out'
    result = open(result_path, 'r')

    test_src_line = test_src.readline()
    result_line = result.readline()

    test_label = []
    result_label = []

    while test_src_line and result_line:
        [tmp1, tmp2, label] = test_src_line.split(' \t')
        test_label.append(int(label[0]))

        result_label.append(int(result_line[0]))

        test_src_line = test_src.readline()
        result_line = result.readline()

    test_src.close()
    result.close()

    count = 0
    for i in range(0, len(test_label)):
        if test_label[i] == result_label[i]:
            count += 1
    # compute precision
    precision = count*1.0/len(test_label)

    # compute score
    ntest_label = numpy.array(test_label)
    nresult_label = numpy.array(result_label)

    score = numpy.sqrt(sum((ntest_label-nresult_label)*(ntest_label-nresult_label)))

    print "Precision = ", precision
    print "Score = ", score
        
