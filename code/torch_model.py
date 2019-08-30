import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import util

class VCNN(nn.Module):
        def __init__(self, Kernal=10, classes=2, embed_size=64):
            super(VCNN, self).__init__()
            self.loss = nn.CrossEntropyLoss()		
            self.embed = nn.Embedding(10000, embed_size)

            self.conv3 = nn.Conv2d(1, Kernal, (2, embed_size), padding=(1, 0))		
            self.final = nn.Linear(Kernal, classes)

        def load_embedding(self, model):
            weight = model.vectors
            self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

        def loss_function(self):
            return self.loss

        def forward(self, x_):
            x_fp = self.embed(x_).unsqueeze(1)
            x3 = F.relu(self.conv3(x_fp)).squeeze(3)
            x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
            cnn_out = x3

            output = self.final(cnn_out)
            return output


class CNN(nn.Module):
	def __init__(self, Kernal=10, classes=2, embed_size=64):
		super(CNN, self).__init__()
		self.loss = nn.CrossEntropyLoss()		
		self.embed = nn.Embedding(10000, embed_size)
		
		self.conv3 = nn.Conv2d(1, Kernal, (3, embed_size), padding=(1, 0))
		self.conv4 = nn.Conv2d(1, Kernal, (4, embed_size), padding=(2, 0))
		self.conv5 = nn.Conv2d(1, Kernal, (5, embed_size), padding=(2, 0))
		
		self.final = nn.Linear(3*Kernal, classes)

	def load_embedding(self, model):
                weight = model.vectors
                self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

	def loss_function(self):
		return self.loss

	def forward(self, x_):
		x_fp = self.embed(x_).unsqueeze(1)

		x3 = F.relu(self.conv3(x_fp)).squeeze(3)
		x4 = F.relu(self.conv4(x_fp)).squeeze(3)
		x5 = F.relu(self.conv5(x_fp)).squeeze(3)

		x3 = F.max_pool1d(x3, x3.size(2)).squeeze(2)
		x4 = F.max_pool1d(x4, x4.size(2)).squeeze(2)
		x5 = F.max_pool1d(x5, x5.size(2)).squeeze(2)

		cnn_out = torch.cat((x3, x4, x5), 1)

		output = self.final(cnn_out)
		return output

class MultiHeadAtt(nn.Module):
	def __init__(self, n_head=2, embed_size=16, d_head=2, classes=2, max_len=256):
		super(MultiHeadAtt, self).__init__()
		self.loss = nn.CrossEntropyLoss()
		self.embed = nn.Embedding(10000, embed_size)
		
		self.n_head = n_head
		self.d_model = embed_size
		self.d_head = d_head
		self.max_len = max_len

		self.q_net = nn.Linear(embed_size, n_head*d_head, bias=False)
		# (seq ,embed) * [(embed, (q1~q2))] = (seq, (q1~q2))
		
		self.kv_net = nn.Linear(embed_size, 2*n_head*d_head, bias=False)
		# (seq, (k1~k2) - (v1~v2))

		self.o_net = nn.Linear(n_head*d_head, embed_size, bias=False)

		self.scale = 1/(d_head ** 0.5)
		self.drop = nn.Dropout(p=0.2)

		self.final = nn.Linear(embed_size*max_len, classes)

	def load_embedding(self, model):
                weight = model.vectors
                self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

	def loss_function(self):
		return self.loss

	def forward(self, x_):
                
                x_fp = self.embed(x_)
                head_q = self.q_net(x_fp)
                head_k, head_v = torch.chunk(self.kv_net(x_fp), 2, -1) # reversed-cat
                
                head_q = head_q.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
                head_k = head_k.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
                head_v = head_v.view(x_fp.size(0), x_fp.size(1), self.n_head, self.d_head)
                # 3dim to 4 dim

                att_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
                # count the attention score 
                att_score.mul_(self.scale)
                # Shrink the size
                att_prob = F.softmax(att_score, dim=1)
                # counting the attention weight.                
                att_vec = torch.einsum('ijbn, jbnd->ibnd', (att_prob, head_v))
                # get the final value
                
                att_vec = att_vec.contiguous().view(att_vec.size(0), att_vec.size(1), self.n_head*self.d_head)

                att_out = self.o_net(att_vec) + x_fp #??
                #word embedding + attention embedding?!

                output = self.final(att_out.view(-1, self.d_model*self.max_len))
                # flatten sum of(attention embedding & word embeddings)
                return output


class MLP(nn.Module):
	def __init__(self, dim=64, classes=2):
		super(MLP, self).__init__()
		self.loss = nn.CrossEntropyLoss() #NLLLOSS + Log_softmax
		
		self.embed = nn.Embedding(10000, dim)
		self.in_net1 = nn.Linear(dim, 16)
		self.in_net2 = nn.Linear(16, 32)
		self.in_net3 = nn.Linear(32, 64)
		self.in_net4 = nn.Linear(64, 128)
		self.o_net = nn.Linear(128, classes)

	def load_embedding(self, model):
                weight = model.vectors
                self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

	def loss_function(self):
		return self.loss
	
	def forward(self, X):
                x_fp = self.embed(X)

                x_fp = F.avg_pool2d(x_fp, (x_fp.size(1), 1)).squeeze(1)
                # doc vector represented by avg.
		
                x_fp = F.relu(self.in_net1(x_fp))
                x_fp = F.relu(self.in_net2(x_fp))
                x_fp = F.relu(self.in_net3(x_fp))
                x_fp = F.relu(self.in_net4(x_fp))
	
                out = self.o_net(x_fp)
                return out

class LSTM(nn.Module):
        def __init__(self, embed_dim=64, hidden_dim=6, classes = 2):
                super(LSTM, self).__init__()
                self.loss = nn.CrossEntropyLoss()
                self.embed = nn.Embedding(10000, embed_dim)

                self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first = True)
                self.final = nn.Linear(hidden_dim, classes)

                self.h_dim = hidden_dim

        def load_embedding(self, model):
                weight = model.vectors
                self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(weight))

        def loss_function(self):
                return self.loss
	
        def forward(self, x_): # (batch, length)
                x_fp = self.embed(x_)   # (batch, length, embed_dim)

                lstm_out, _ = self.lstm(x_fp)
                # input size: (bathc, length, embed_dim)
                # O, h_s, c_s: (batch, length, hidden_dim)

                #print(lstm_out.size())
                out = self.final(lstm_out[:, -1, :])

                return out

class GCN(nn.Module):
    def __init__(self, embed_dim=64, nhid=3, nclass=2, dropout=0.2):
        super(GCN, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        
        self.gc1 = GraphConvolution(embed_dim, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def loss_function(self):
        return self.loss

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        out = self.gc2(x, adj)
        return out
