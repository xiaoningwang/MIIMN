from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.aspect_attention import Aspect_Attention, Aspect_Attention_v2
from layers.sentiment_attention import Sentiment_Attention, Sentiment_Attention_v2
from layers.acd_prediction_layer import Aspect_Category_Prediction, Aspect_Category_Prediction_v2, Aspect_Category_Prediction_v3
from layers.asc_prediction_layer import Shared_Sentiment_Prediction, Shared_Sentiment_Prediction_v2, Shared_Sentiment_Prediction_v3
import torch
import torch.nn as nn


class JointModel_v1(nn.Module):
    '''
    v1: 遵循Baidu A-Joint-Model论文思路，输入一条样本，同时输出所有aspect的ACD/ASC预测结果。
    '''
    def __init__(self, embedding_matrix, opt):
        super(JointModel_v1, self).__init__()
        self.opt = opt
        
        # 1. embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # 2. bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # 3. aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        self.aspect_attention_X = Aspect_Attention(opt.embed_dim, opt.hidden_dim, opt.aspect_num)
        self.aspect_attention_H = Aspect_Attention(opt.hidden_dim*2, opt.hidden_dim, opt.aspect_num)
        
        # 4. aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction(opt.embed_dim, opt.hidden_dim*2, opt.aspect_num)
        
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        four elements of inputs:
            1. sentence token ids
            2. aspect word ids
            3. img features
            4. img nums
        '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len); 句子不定长，统一[PAD]到相同长度，[PAD]对应id为0
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度      
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch,max_seq_len, hidden_dim*2)
        
        text_raw_emb = self.dropout(text_raw_emb)
        text_hidden_vec = self.dropout(text_hidden_vec)
        
        v_X_aspect = self.aspect_attention_X(text_raw_emb)  # v_X_Aj, (batch, aspect_num, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec)  # v_H_Aj, (batch, aspect_num, hidden_dim*2)
        
        # acd prediction layer, output = (batch, aspect_num, 1)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect)
        
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, aspect_num, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, aspect_num, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, aspect_num, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits


class JointModel_v2(nn.Module):
    '''
    v2: 适应sample-aspect的pair形式样本，输入一条pair样本，输出关于指定aspect的ACD/ASC预测结果。
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(JointModel_v2, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # 1. embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # 2. bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # 3. aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        # self.aspect_attention_X = Aspect_Attention(opt.embed_dim, opt.hidden_dim, opt.aspect_num)
        # self.aspect_attention_H = Aspect_Attention(opt.hidden_dim*2, opt.hidden_dim, opt.aspect_num)
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # 4. aspect category prediction layer
        # self.acd_pred_layer = Aspect_Category_Prediction(opt.embed_dim, opt.hidden_dim*2, opt.aspect_num)
        self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        two elements of inputs: sentence token ids、aspect ids
        '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        text_raw_emb = self.dropout(text_raw_emb)
        text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer

        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect, aspect_ids)
        
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits


class JointModel_simplified(nn.Module):
    '''
    simplified: 基于v2版本，删除ACD/ASC子网络中使用v_X_aspect信息的路径，仅保留v_H_aspect一路。
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(JointModel_simplified, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # 1. embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # 2. bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # 3. aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        # self.aspect_attention_X = Aspect_Attention(opt.embed_dim, opt.hidden_dim, opt.aspect_num)
        # self.aspect_attention_H = Aspect_Attention(opt.hidden_dim*2, opt.hidden_dim, opt.aspect_num)
        # self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # 4. aspect category prediction layer
        # self.acd_pred_layer = Aspect_Category_Prediction(opt.embed_dim, opt.hidden_dim*2, opt.aspect_num)
        # self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v3(opt.hidden_dim*2, opt.hidden_dim, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        two elements of inputs: sentence token ids、aspect ids
        '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        text_raw_emb = self.dropout(text_raw_emb)
        text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer

        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_H_aspect, aspect_ids)
        
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits
