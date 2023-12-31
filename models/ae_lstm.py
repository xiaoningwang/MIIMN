# -*- coding: utf-8 -*-
# file: lstm.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.


from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn


class AELSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(AELSTM, self).__init__()
        self.opt = opt
        self.n_head = 1
        self.embed_dim = opt.embed_dim
        
        if opt.dataset != 'MASAD' or (opt.dataset == 'MASAD' and opt.shared_emb == 1):
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        elif opt.dataset == 'MASAD' and opt.shared_emb == 0:
            word_embedding_matrix, aspect_embedding_matrix = embedding_matrix[0], embedding_matrix[1]
            self.embed = nn.Embedding.from_pretrained(torch.tensor(word_embedding_matrix, dtype=torch.float), freeze=False)
            self.aspect_embed = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float), freeze=False)
        
        self.lstm = DynamicLSTM(opt.embed_dim*2, opt.hidden_dim, num_layers=1, batch_first=True)
        self.attention = Attention(opt.hidden_dim, score_function='mlp')
        
        if opt.embed_dim != opt.hidden_dim:  # asp_squ: (batch, embed_dim) ---> (batch, hidden_dim)
            self.transform = nn.Linear(opt.embed_dim, opt.hidden_dim)
        else:
            self.transform = None
        
        self.dense = nn.Linear(opt.hidden_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_raw_indices = inputs[0]  # (batch, common_review_length); 句子不定长，统一[PAD]到相同长度，[PAD]对应id为0
        aspect_indices = inputs[1]  # (batch, common_aspect_length);  Aspect不定长，统一[PAD]到相同长度，[PAD]对应id为0
        
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)
        
        # 1. generate review embeddings, (batch, common_review_length, emb_dim)
        x = self.embed(text_raw_indices)
        
        # 2. generate aspect embeddings = the average of all aspect word embedding
        # shape = (batch, common_aspect_length, emb_dim)
        if self.opt.dataset != 'MASAD' or (self.opt.dataset == 'MASAD' and self.opt.shared_emb == 1):
            aspect = self.embed(aspect_indices)  
        elif self.opt.dataset == 'MASAD' and self.opt.shared_emb == 0:
            aspect = self.aspect_embed(aspect_indices)
        aspect = torch.sum(aspect, dim=1)  # (batch, emb_dim)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))  # do averaging
        asp_squ = aspect.unsqueeze(dim=1)  # (batch, 1, emb_dim)
        
        # 3. concat each word embedding in a review with related aspect embedding
        asp_re = asp_squ.repeat(1, x.size()[1], 1)  # (batch, common_review_length, emb_dim)
        asp_x = torch.cat((x, asp_re), dim=-1)
        
        # 4. 将asp_x输入单向LSTM，计算每条评论中的每个单词的隐藏层输出，作为text_memory
        text_memory, (_, _) = self.lstm(asp_x, x_len)
        
        # 5. 以aspect word为query，对lstm输出的text memory做Attention，提取aspect相关信息
        # Note：请注意，attention层默认输入的key和query向量维度一致，但是这里输入的text_memory和asp_squ的维度是不一致的
        # text_memory的维度是LSTM的隐层维度hidden_dim = 100，asp_squ的维度是初始的word embedding dim = 300
        if self.transform is not None:
            asp_squ = self.transform(asp_squ)  # (batch, 1, emb_dim) --> (batch, 1, hidden_dim)
        out_at = self.attention(text_memory, asp_squ).squeeze(dim=1)  # (batch, 1, hidden_dim) --> (batch, hidden_dim)
        
        # 6. 预测层
        out_at = out_at.view(out_at.size(0), -1)
        out = self.dense(out_at)
        return out
