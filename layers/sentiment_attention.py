import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sentiment_Attention(nn.Module):
    def __init__(self):
        ''' 
        Sentiment_Attention without new parameters
        '''
        super(Sentiment_Attention, self).__init__()

    def forward(self, k, q):
        '''
        k: (batch, text_len, vector_dim)
        q: (batch, aspect_num, vector_dim)
        
        k为content embedding矩阵
        q为aspect attention向量
        
        目标是用每个aspect的attention向量作为query向量，从content embedding中为各个aspect抽取语义信息
        最终产生(batch, aspect_num, vector_dim)的输出
        '''
        vec_dim = k.shape[-1]
        
        att_scores = torch.bmm(q, k.permute(0,2,1))  # (batch, aspect_num, text_len)
        att_scores = torch.div(att_scores, math.sqrt(vec_dim))
        att_weights = F.softmax(att_scores, dim = -1)
        
        output = torch.bmm(att_weights, k)  # (batch, aspect_num, vector_dim)
        
        return output


class Sentiment_Attention_v2(nn.Module):
    '''
    version2 is designed for pair sample.
    '''
    def __init__(self):
        ''' 
        Sentiment_Attention without new parameters
        '''
        super(Sentiment_Attention_v2, self).__init__()

    def forward(self, k, q):
        '''
        k: (batch, text_len, vector_dim)
        q: (batch, vector_dim)
        
        k为content embedding矩阵, q为aspect attention向量
        '''
        vec_dim = k.shape[-1]
        
        att_scores = torch.bmm(k, q.unsqueeze(dim=-1)).squeeze()  # (batch, text_len)
        att_scores = torch.div(att_scores, math.sqrt(vec_dim))
        att_weights = F.softmax(att_scores, dim = -1)
        
        output = torch.bmm(att_weights.unsqueeze(dim=1), k).squeeze()  # (batch, vector_dim)
        
        return output

if __name__ == '__main__':
    k = torch.FloatTensor(3, 2, 4)  # (batch, text_len, vector_dim)
    q = torch.FloatTensor([0 for i in range(36)]).resize(3, 3, 4)  # (batch, aspect_num, vector_dim)
    
    sentiment_attention_layer = Sentiment_Attention()
    
    output = sentiment_attention_layer(k, q)