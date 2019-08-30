from Executer import executer
from Handler import handler
import torch_model as models
import numpy as np
from Dataset import iclr_data, rt_data
from Encoder import OneHotEncoder as ohe
import util
from w2v import w2v_paper, w2v_pretrained

arg = [30, 64, 'cpu']

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
#w2v = w2v_pretrained('word2vec-google-news-300')
#w2v = w2v_pretrained('glove-wiki-gigaword-100')
#w2v = w2v_pretrained('paper6k-300d') # which in the kv repo

testname ='paper6k-100d-sw'
#w2v_paper(testname)
#w2v = w2v_pretrained(testname)

#Selftrained model(in gensim)
#---------------
#w2v = w2v_selftrained(x, lbl, oc = False)


#Split (pretrained v.s. no pretrained)
#---------------
d = 100
model = False

#model = w2v.model
#d = w2v.model.vector_size
#x_encoded = ohe().fit_transform_pretrained(x, w2v.getDict())
x_encoded = ohe().fit_transform(x)  #encoded with wntire dataset.

#loaders = handler().split_tvt((x_encoded, lbl))
set_loaders = handler().skfold(x_encoded, lbl, 10)


def run_nn(model, loaders, model_name, prt_model = False):
    print('==='+ model_name +'===<start>')
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of para: {}'.format(total))

    tr, val, te = loaders
    
    #Execute model_1
    #train and validate
    e = executer(model, model_name, arg, prt_model)
    e.train(tr, val)
    
    #testnn.Embedding.from_pretrained(weight)
    e.validate(te)
    print('Accuracy: {}'.format(e.hist['val_acc']))
    print('F1-Score: {}'.format(e.f1))
    print(model)

def run_nn_cv(model, set_loaders, model_name, prt_model = False):
    print('==='+ model_name +'===<start>')
    
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('# of para: {}'.format(total))

    trset, valset, teset = set_loaders

    e = executer(model, model_name, arg, prt_model)
    cv_score, f1 = e.cv(trset, valset, teset)
    print(model)
    print(np.mean(cv_score))
    print(np.mean(f1))

run_nn_cv(models.MLP(d, 2), set_loaders, 'MLP.pt', model)
run_nn_cv(models.VCNN(10, 2, d), set_loaders, 'VCNN.pt', model)
run_nn_cv(models.CNN(10, 2, d), set_loaders, 'CNN.pt', model)
run_nn_cv(models.LSTM(d, 6, 2), set_loaders, 'LSTM.pt', model)
run_nn_cv(models.MultiHeadAtt(2, d, 8, 2), set_loaders, 'attention.pt', model)
##run_nn_cv(models.GCN(d, 3, 2), loaders, 'GCN.pt', model)

##run_nn(models.MLP(d, 2), loaders, 'MLP.pt', model)
##run_nn(models.VCNN(10, 2, d), loaders, 'VCNN.pt', model)
##run_nn(models.CNN(10, 2, d), loaders, 'CNN.pt', model)
##run_nn(models.LSTM(d, 6, 2), loaders, 'LSTM.pt', model)
##run_nn(models.MultiHeadAtt(2, d, 8, 2), loaders, 'attention.pt', model)
##run_nn(models.GCN(d, 3, 2), loaders, 'GCN.pt', model)
