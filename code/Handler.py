import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data  # Dataset format in pytorch
from torch.autograd import Variable  #?? meaning of variable.
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split as tts

import math
import numpy as np
import time
import util
from statistics import mode, mean

'''
Usage:
    1. Split the raw dataset(which include train/validate/test sets).
    2. Load each set into dataloader(for pytorch), or export the dataset for sklearn.
    3. TF not supported yet.
'''

class handler():

    def __init__(self, arg = [300, 64, 'cpu']):
        self.n_batch = arg[1]
        self.max_length = 256
    
    def split(self, data, seg_size = (0.7, 0.1, 0.2), mode = 'pytorch'): #MUST shuffled

        x_all, lbl_all = data #entire

        x_train, x_test, lbl_train, lbl_test = \
            tts(x_all, lbl_all, stratify = lbl_all, test_size = 0.2, random_state = 87)

        #length = [len(l) for l in x_train]
        #self.max_len = int(max(length))

        
        if mode == 'sklearn':
            return x_train, x_test, lbl_train, lbl_test

        else:  #mdoe == pytorch
            #x_train, lbl_train = self.resample(x_train, lbl_train, mode = 'ros')

            x_train, x_val, lbl_train, lbl_val = \
                tts(x_train, lbl_train, test_size = 1/8, stratify = lbl_train, random_state = 87)

            # An resampling issue on validatino or not could be considered.
            self.train = x_train, lbl_train

            #|____train____|_val_|__test__|
            train_loader = self.load(x_train, lbl_train)
            val_loader = self.load(x_val, lbl_val)
            test_loader = self.load(x_test, lbl_test)
            #Get the index of each set

            return train_loader, val_loader, test_loader


    def load(self, x_, lbl_, dtype = [torch.LongTensor, torch.LongTensor]):

        x_ = util.padding(x_, 256) # fill the x with same length to train the NN model.

        #TENSORIZE...the data.
        feature = Variable(dtype[0](x_))
        label = Variable(dtype[1](lbl_))
                
	#COMBINE...into a Dataset obj.
        dataset = Data.TensorDataset(feature, label)

        print("Data size: ", len(dataset))

        #PACK...into DataLoader obj.(which can implement batching feed)
        data_loader = Data.DataLoader(dataset, self.n_batch)

        return data_loader

    def resample(self, x, y, mode = 'ros'):
        #Resampling(over/undersampling)
        if mode == 'ros':
            x_re, y_re = util.oversampling(x.reshape(-1, 1), y)
        elif mode == 'rus':
            x_re, y_re = util.undersampling(x.reshape(-1, 1), y)
        print('RESAMPLED!')
        
        return x_re.reshape(-1, ), y_re


        
