import gensim
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts, get_tmpfile
from Dataset import nips_data, icml_data
import util
import numpy as np
from sklearn.utils import shuffle
#https://arxiv.org/pdf/1507.05523.pdf

class w2v_paper():

    def __init__ (self, x_new = None):
        self.model = None
        x = self._collect(x_new)
        x = shuffle(x, random_state = 87) # shuffle is proved to be perform better in embedding.
        self.train(self._np2list(x))

    def _np2list(self, np_x):
        x = []
        for doc in np_x:
            doclist = [word for word in doc]
            x.append(doclist)
        return x
        
    def _collect(self, x_new = None):
        paper = np.empty([0, ])
        for nips in range(13, 19):
            paper = np.concatenate((paper, nips_data(nips).getAbstract()))
        for icml in range(17, 20):
            paper = np.concatenate((paper, icml_data(icml).getAbstract()))
        if x_new is not None :
            paper = np.concatenate((paper, x_new)) #2d-array
            
        return paper

    def train(self, x, size_ = 100, window_ = 5, iter_ = 20):
        self.model = Word2Vec(x, size = size_, window = window_, min_count=1, iter=iter_)

    def save(self, fname):
        wv = self.model.wv
        wv.save('kv/'+fname)
        print('saved!')
