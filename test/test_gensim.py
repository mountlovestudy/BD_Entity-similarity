from gensim import *
import numpy

if __name__ == '__main__':
    lsi = models.LsiModel.load('../result/lsimodel.txt')
    corpus = corpora.MmCorpus('../result/doc_wordid.txt')
    # index = similarities.MatrixSimilarity(lsi[corpus])
    # index.save('../result/similarity.index')
    index = similarities.MatrixSimilarity.load('../result/similarity.index')
    corpus_first=list(corpus)[4308]
    vec_lsi=lsi[corpus_first]
    sims=index[vec_lsi]
    # sort(sims)
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    for s in sort_sims:
        if s[0] != 4267:
            print s
        else:
            break
    # print(list(enumerate(sims)))
