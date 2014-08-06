#!/usr/bin/python

"""Pre-processing the json data, get two dicts from entity.json, one is words-id, the other is id-num-mapping.

   entity_words_id_dict, id_num_mapping = generate_dict(entity_file_path, stop_words_path)
"""

__authors__ = [
    '"Zhao Yalong" <yalong85841@gmail.com>'
]

import json
import jieba
import codecs
import linecache
import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from read_and_write import write_dict_into_GB18030, get_stop_words, write_dict_into_utf8

def generate_dict(filepath1, filepath2):
    f = open(filepath1)
    stopwords = get_stop_words(filepath2)
    # lines = f.readlines()
    words_id_dict = {}
    id_num_mapping = {}
    cnt = 0
    for line in f.readlines():
    # for i in xrange(1):
        # load json
        # line = linecache.getline(filepath1, 2990)
        entity = line.split(' \t')
        entity_id = entity[0]
        entity_json = json.loads(entity[1].decode("GB18030"), encoding = "GB18030", strict=False)
        # entity_json = json.loads(entity[1])

        for value in entity_json.values():
            for item in value:
                seg_list = jieba.cut(item.encode("GB18030"))
                for seg in seg_list:
                    if seg + '\n' not in stopwords:
                        if seg in words_id_dict.keys():
                            words_id_dict[seg].append(cnt)
                        else:
                            words_id_dict[seg] = [cnt]

        # get id_num_mapping
        id_num_mapping[entity_id] = cnt
        print cnt
        line = f.readline()
        cnt = cnt + 1
    f.close()
    return words_id_dict, id_num_mapping
        
if __name__ == '__main__':
    entity_file_path = '../data/entity.json'
    stop_words_path = '../data/stopwords.txt'
    linecache.clearcache()
    entity_words_id_dict, id_num_mapping = generate_dict(entity_file_path, stop_words_path)

    # write result in json file
    words_id_dict_path = '../result/words_id_dict.json'
    write_dict_into_utf8(words_id_dict_path, entity_words_id_dict)
    write_dict_into_GB18030(words_id_dict_path, entity_words_id_dict)
    
    id_num_mapping_path = '../result/id_num_mapping.json'
    write_dict_into_utf8(id_num_mapping_path, id_num_mapping)
    write_dict_into_GB18030(id_num_mapping_path, id_num_mapping)
    
    
