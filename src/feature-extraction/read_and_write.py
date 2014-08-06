#!/usr/bin/python

"""Read and write file encode with utf8 and GB18030.

get_stop_words(filepath):
write_dict_into_GB18030(path, dic):
write_dict_into_utf8(path, dic)
"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import json
import codecs
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

def get_stop_words(filepath):
    f = codecs.open(filepath, 'r')
    line = f.readline()
    stopwords = []
    while line:
        stopwords.append(line.decode("GB18030"))
        #stopwords.append(line.decode("UTF-8"))
        line = f.readline()
    return stopwords
    
def write_dict_into_GB18030(path, dic):
    """
    [yas] elisp error! Symbol's value as variable is void: text
    """
    dict_json = json.dumps(dic, ensure_ascii = False)
    f = codecs.open(path, 'w')
    f.write(dict_json.encode('GB18030'))
    f.close()

def write_dict_into_utf8(path, dic):
    """
    [yas] elisp error! Symbol's value as variable is void: text
    """
    dict_json = json.dumps(dic) #, ensure_ascii = False)
    f = codecs.open(path, 'w')
    f.write(dict_json) # .encode('GB18030'))
    f.close()
