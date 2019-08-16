from Executer import executer
from Handler import handler
import torch_model as models
import numpy as np
from Dataset import iclr_data, rt_data
from Encoder import OneHotEncoder as ohe
import util
from w2v import w2v_paper

arg = [300, 64, 'cpu']

#Prepare Dataset
#---------------
OR = iclr_data(18, 'submission')
lbl18 = OR.getY()
x18 = OR.getAbstract()
x18_t = OR.getTitle()

OR = iclr_data(17, 'submission')
lbl17 = OR.getY()
x17 = OR.getAbstract()
x17_t = OR.getTitle()

x = np.concatenate((x18, x17))
lbl = np.concatenate((lbl18, lbl17))

#Load Pretrained Vocabulary
#---------------
from gensim.models import KeyedVectors, Word2Vec
import gensim.downloader as api
util.hackssl()
#w2v = api.load('word2vec-google-news-300')
#w2v = api.load('glove-wiki-gigaword-100')
#w2v = KeyedVectors.load('kv/paper6k-300d.kv')   #paper 6k
#w2v = w2v_paper().model.wv #paper6k-100d

#Selftrained model
#---------------
#w2v = ohe().self_train(x, lbl).wv 
#w2v = ohe().self_train_one(x, lbl).wv  #with one class

#Cheat on OOV
#---------------
zero = np.zeros([1, 300])
w2v.add('<ZERO>', zero)
unk = np.random.randn(1, 300)
w2v.add('<UNK>', unk)
w2v_dict = w2v.vocab


#Split
#---------------
#x_encoded = ohe().fit_transform(x)
x_encoded = ohe().fit_transform_pretrained(x, w2v_dict)
loaders = handler().split((x_encoded, lbl), mode = 'pytorch')




def run_nn(model, loaders, model_name, prt_model = False):
    print('==='+ model_name +'===<start>')
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of para: {}'.format(total))

    #loaders unpack
    tr, val, te = loaders
    
    #Execute model_1
    #train and validate
    e = executer(model, model_name, arg, prt_model)
    e.train(tr, val)
    
    #testnn.Embedding.from_pretrained(weight)
    e.validate(te)
    print('==='+ model_name +'===<end>')
    print('Accuracy: {}'.format(e.hist['val_acc']))
    print('F1-Score: {}'.format(e.f1))

d = 300
#w2v = False

run_nn(models.MLP(d, 2), loaders, 'MLP.pt', w2v)
run_nn(models.CNN(10, 2, d), loaders, 'CNN.pt', w2v)
run_nn(models.LSTM(d, 6, 2), loaders, 'LSTM.pt', w2v)
run_nn(models.MultiHeadAtt(2, d, 2, 2), loaders, 'attention.pt', w2v)

