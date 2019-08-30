from Executer import executer
from Handler import handler
import torch_model as models
import numpy as np
from Dataset import iclr_data, rt_data
from Encoder import OneHotEncoder as ohe
import util
from w2v import w2v_paper, w2v_pretrained
import torch.utils.data
from torch.autograd import Variable
from skorch import NeuralNet

OR = iclr_data(18, 'submission')
lbl18 = OR.getY()
x18 = OR.getAbstract()
x18_t = OR.getTitle()

OR = iclr_data(17, 'submission')
lbl17 = OR.getY()
x17 = OR.getAbstract()
x17_t = OR.getTitle()

x = np.concatenate((x18_t, x17_t))
lbl = np.concatenate((lbl18, lbl17))
x_encoded = ohe().fit_transform(x)  #encoded with wntire dataset.


class torch_cv(torch.utils.data.Dataset):
    def __init__(self, x_, lbl_, dtype = [torch.LongTensor, torch.LongTensor]):

        x_ = util.padding(x_, 256)
        
        feature = Variable(dtype[0](x_))
        label = Variable(dtype[1](lbl_))
        
        self.X = feature
        self.Y = label

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)

train_ds = torch_cv(x_encoded, lbl)
nn = models.MLP(100, 2)
net = NeuralNet(nn, criterion=nn.loss_function())
net.fit(x_encoded, lbl)
