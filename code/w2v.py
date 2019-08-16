import gensim
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader as api
from gensim.test.utils import common_texts, get_tmpfile
from Dataset import nips_data, icml_data
import util
import numpy as np
from sklearn.utils import shuffle
#https://arxiv.org/pdf/1507.05523.pdf

class w2v_paper():

    def __init__ (self, f_name = 'paper6k-test', x_new = None):
        x = self._collect(x_new)
        x = shuffle(x, random_state = 1) # shuffle is proved to be perform better in embedding.
        x = util.np2list(x)
        self.model = Word2Vec(x, size = 100, window = 5, min_count = 1, iter = 20)
        self.save(f_name)

    def _collect(self, x_new = None):
        paper = np.empty([0, ])
        for nips in range(13, 19):
            paper = np.concatenate((paper, nips_data(nips).getAbstract()))
        for icml in range(17, 20):
            paper = np.concatenate((paper, icml_data(icml).getAbstract()))
        if x_new is not None :
            paper = np.concatenate((paper, x_new)) #2d-array
        return paper

    def save(self, fname):
        self.model.save('kv/'+fname+'.kv')
        print('saved!')



class w2v_pretrained():

    def __init__(self, model_name):        
        try:
            util.hackssl()
            self.model = api.load(model_name) # import the gensim pretrained model
            # 'word2vec-google-news-300', 'glove-wiki-gigword-100d'
        except:
            self.model = KeyedVectors.load('kv/'+model_name+'.kv')
            # 'paper6k-300d', 'paper6k-100d'
        self.add()
        
    def getDict(self):
        return self.model.vocab

    def add(self, token=None, vector=None):
        self.model.add('<ZERO>', np.zeros([1, self.model.vector_size]))
        self.model.add('<UNK>', np.random.randn(1, self.model.vector_size))
        #default input the <zero> & <unk>
        if token != None:
                self.model.add(token, vector)


class w2v_selftrained(): # need to be pre-split,

    def __init__(self, corpus, lbl, oc = False):
        x, _, lbl, _ = handler().split_tt((corpus, lbl))
        if oc == True:
            fil = (lbl == 1)
            x = x[fil]
            print('One class strategy.')
        x = util.np2list(x)
        self.model = Word2Vec(x, size = 100, window = 5, min_count = 1, iter = 20)
        self.add()

    def getDict():
        return self.model.vocab

    def add(self, token=None, vector=None):
        self.model.add('<ZERO>', np.zeros([1, self.model.vector_size]))
        self.model.add('<UNK>', np.random.randn(1, self.model.vector_size))
        #default input the <zero> & <unk>
        if token != None:
                self.model.add(token, vector)
