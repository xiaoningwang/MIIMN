# -*- coding: utf-8 -*-
# file: ram.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn
import torchvision.models as Models

class MIMN(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1-float(idx)/int(memory_len[i]))
        return memory

    def __init__(self, embedding_matrix, opt):
        super(MIMN, self).__init__()
        self.opt = opt
        
        # Image Embedding Layer: 基于预训练的ResNet18/50模型，取倒数第二层的输出作为image embedding
        self.img_extractor = nn.Sequential(*list(Models.resnet18(pretrained=True).children())[:-1])
        
        # Text Embedding Layer: 读取预构造的embedding矩阵来初始化embedding层, word embedding跟随模型一起训练
        if opt.dataset != 'MASAD' or (opt.dataset == 'MASAD' and opt.shared_emb == 1):
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        elif opt.dataset == 'MASAD' and opt.shared_emb == 0:
            word_embedding_matrix, aspect_embedding_matrix = embedding_matrix[0], embedding_matrix[1]
            self.embed = nn.Embedding.from_pretrained(torch.tensor(word_embedding_matrix, dtype=torch.float), freeze=False)
            self.aspect_embed = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float), freeze=False)
        
        # Bi-LSTM for review/aspect/image
        # batch_first==True显式地声明输入数据的第一个维度为batch_size，比如(100,10,3)表示有100个batch，每个batch有长度为10的序列
        # 同时保证LSTM的输出结果的第一维也是batch_size，即形状为(batch_size, seq_len, hidden_size)
        # 但是，batch_first==True不保证hn和cn的第一维表示batch
        self.bi_lstm_context = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_aspect = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.bi_lstm_img = DynamicLSTM(opt.embed_dim_img, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Attention Layer
        # according to this code, 对于text/image侧来说, 不同hop的Attention操作都是使用同一个Attention层
        self.attention_text = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.attention_img = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.attention_text2img = Attention(opt.hidden_dim * 2, score_function='mlp')
        self.attention_img2text = Attention(opt.hidden_dim * 2, score_function='mlp')
        
        # GRU for text and image
        self.gru_cell_text = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        self.gru_cell_img = nn.GRUCell(opt.hidden_dim*2, opt.hidden_dim*2)
        
        # Batch Normalization (do not use BN and Dropout together)
        self.bn = nn.BatchNorm1d(opt.hidden_dim*4, affine=False)
        
        # Dropout Layer (do not use BN and Dropout together)
        self.dropout = nn.Dropout(p=opt.dropout)
        
        # Prediction Layer
        self.fc = nn.Linear(opt.hidden_dim * 4, opt.polarities_dim)

    def forward(self, inputs):
        # four input elements for MIMN: review token ids, aspect token ids, review imgs, img nums
        text_raw_indices = inputs[0]  # (batch, max_seq_len). 句子不定长，统一[PAD]到max_seq_len，[PAD]对应emb idx为0
        aspect_indices = inputs[1]  # (batch, max_aspect_len). Aspect不定长，统一[PAD]到max_aspect_len，[PAD]对应emb idx为0
    
        imgs = inputs[2]  # (batch, max_img_len, img_emb_dim)
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)

        num_imgs = inputs[3]  # batch中每个评论对应的图像真实数量
        
        text_memory_len = torch.sum(text_raw_indices != 0, dim=-1)  # (batch, ). 去除[PAD], 计算每个review text的真实长度
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)  # (batch, ). 去除[PAD], 计算每个aspect word的真实长度
        imgs_memory_len = torch.tensor(num_imgs).to(self.opt.device)  # (batch, )
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.opt.device)  # 等价于aspect_len
        
        text_raw = self.embed(text_raw_indices)  # review word embeddings
        if self.opt.dataset != 'MASAD' or (self.opt.dataset == 'MASAD' and self.opt.shared_emb == 1):
            aspect = self.embed(aspect_indices)  # aspect word embeddings
        elif self.opt.dataset == 'MASAD' and self.opt.shared_emb == 0:
            aspect = self.aspect_embed(aspect_indices)  # aspect word embeddings

        # 双向LSTM处理评论序列/Aspect序列/图像序列
        text_memory, (_, _) = self.bi_lstm_context(text_raw, text_memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        img_memory, (_, _) = self.bi_lstm_img(imgs, imgs_memory_len)
        
        # do average pooling for asepct word embeddings to get aspect representation
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et_text = aspect
        et_img = aspect

        for _ in range(self.opt.hops):
            it_al_text2text = self.attention_text(text_memory, et_text).squeeze(dim=1)
            it_al_img2text = self.attention_img2text(text_memory, et_img).squeeze(dim=1)
            it_al_text = (it_al_text2text + it_al_img2text)/2
            # it_al_text = it_al_text2text

            it_al_img2img = self.attention_img(img_memory, et_img).squeeze(dim=1)
            it_al_text2img = self.attention_text2img(img_memory, et_text).squeeze(dim=1)
            it_al_img = (it_al_img2img + it_al_text2img)/2
            # it_al_img = it_al_img2img

            et_text = self.gru_cell_text(it_al_text, et_text)
            et_img = self.gru_cell_img(it_al_img, et_img)
            
        et = torch.cat((et_text, et_img), dim=-1)
        
        # do not use BN and dropout together
        # et = self.bn(et)
        # et = self.dropout(et)
        
        out = self.fc(et)  # (batch, polarity_nums)
        
        return out

