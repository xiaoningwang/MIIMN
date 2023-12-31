import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Aspect_Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, aspect_num = 1):
        '''aspect attention layer
        
        input: X = x1, x2, ..., xn
        output: V = α1*x1 + ... + αn*xn
        
        pipeline:
        1. ui = tanh(W*xi + b), dim(ui) = hidden_dim
        2. αi = softmax(ui*q), dim(q) = hidden_dim
        
        aspect_num <==> head_num
        '''
        super(Aspect_Attention, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.aspect_num = aspect_num
        
        # aspect_num = n
        self.W = nn.Parameter(torch.FloatTensor(aspect_num, embed_dim, hidden_dim))
        self.b = nn.Parameter(torch.FloatTensor(aspect_num, 1, hidden_dim))
        self.q = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim, 1))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        self.q.data.uniform_(-stdv, stdv)
        
    def forward(self, X):
        batch_size = X.shape[0]  # x.shape = (batch, text_len, embed_dim)
        
        XX = X.repeat(self.aspect_num, 1, 1)  # (aspect_num*batch, text_len, embed_dim)
        XX = XX.view(self.aspect_num, -1, self.embed_dim)  # (aspect_num, batch*text_len, embed_dim)
        
        U = torch.bmm(XX, self.W)  # (aspect_num, batch*text_len, hidden_dim) 
        U = torch.add(U, self.b)  
        U = F.tanh(U)
        
        att_scores = torch.bmm(U, self.q).squeeze()  # (aspect_num, batch*text_len)
        att_scores = torch.div(att_scores, math.sqrt(self.hidden_dim))
        att_scores = att_scores.view(self.aspect_num, batch_size, -1).permute(1, 0, 2)  # (batch, aspect_num, text_len)
        
        att_weights = F.softmax(att_scores, dim = -1)
        
        output = torch.bmm(att_weights, X)  # (batch, aspect_num, embed_dim)
        
        return output


class Aspect_Attention_v2(nn.Module):
    def __init__(self, embed_dim, hidden_dim, aspect_num):
        '''
        version2 is designed for pair sample.
        '''
        super(Aspect_Attention_v2, self).__init__()
        
        self.embed_dim = embed_dim  # a.k.a input_dim
        self.hidden_dim = hidden_dim
        self.aspect_num = aspect_num
        
        # aspect_num = n
        #self.W = nn.Parameter(torch.FloatTensor(aspect_num, embed_dim, hidden_dim))
        #self.b = nn.Parameter(torch.FloatTensor(aspect_num, 1, hidden_dim))
        #self.q = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim, 1))
        #self.reset_parameters()
        self.weight_embs = nn.Embedding(self.aspect_num, embed_dim*hidden_dim)
        self.bias_embs = nn.Embedding(self.aspect_num, hidden_dim)
        self.query_embs = nn.Embedding(self.aspect_num, hidden_dim)
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.W.data.uniform_(-stdv, stdv)
        self.b.data.uniform_(-stdv, stdv)
        self.q.data.uniform_(-stdv, stdv)
        
    def forward(self, X, aspect_ids):
        '''
        X：(batch, text_len, embed_dim)
        aspect_ids：(batch, ) ----- torch.LongTensor()
        '''
        batch_size = X.shape[0]  # x.shape = (batch, text_len, embed_dim)
        if len(aspect_ids) == 2:
            aspect_ids = aspect_ids.squeeze()  # (batch, 1) --> (batch, )
        
        W = self.weight_embs(aspect_ids)  # (batch, embed_dim*hidden_dim)
        W = W.view(batch_size, self.embed_dim, self.hidden_dim)  # (batch_size, embed_dim, hidden_dim) 
        
        b = self.bias_embs(aspect_ids)  # (batch, hidden_dim)
        b = b.unsqueeze(dim=1)  # (batch, 1, hidden_dim)
        
        q = self.query_embs(aspect_ids)  # (batch, hidden_dim)
        q = q.unsqueeze(dim=-1)  # (batch, hidden_dim, 1)
        
        #print(X.shape)
        #print(W.shape)

        U = torch.bmm(X, W)  # (batch, text_len, hidden_dim) 
        U = torch.add(U, b)  
        U = F.tanh(U)
        
        att_scores = torch.bmm(U, q).squeeze(dim=-1)  # (batch, text_len)
        att_scores = torch.div(att_scores, math.sqrt(self.hidden_dim))
        
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = att_weights.unsqueeze(dim=1)  # (batch, 1, text_len)
        
        #print(att_weights.shape)
        #print(X.shape)

        output = torch.bmm(att_weights, X).squeeze()  # (batch, embed_dim)
        
        return output
    
if __name__ == '__main__':
    aspect_attention_layer = Aspect_Attention(embed_dim = 3, hidden_dim = 5, aspect_num = 2)
    X = torch.FloatTensor([i for i in range(24)]).resize(2,4,3)
    res = aspect_attention_layer(X)
