from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
import torch
import torch.nn as nn

class Aspect_Prediction(nn.Module):
    def __init__(self, input_dim, polarity_num, aspect_num):
        '''
        only for End2End_LSTM
        '''
        super(Aspect_Prediction, self).__init__()
        
        self.input_dim = input_dim
        self.polarity_num = polarity_num
        self.aspect_num = aspect_num
        
        #self.W1 = nn.Parameter(torch.FloatTensor(aspect_num, input_dim, polarity_num))
        #self.b1 = nn.Parameter(torch.FloatTensor(aspect_num, polarity_num))
        self.W1_embs = nn.Embedding(aspect_num, input_dim*polarity_num)
        self.b1_embs = nn.Embedding(aspect_num, polarity_num)
        
    def forward(self, X, aspect_ids):
        '''
        X: (batch, input_dim)
        aspect_ids：(batch, ) ----- torch.LongTensor()
        '''
        batch_size = X.shape[0]
        if len(aspect_ids.shape) == 2:
            aspect_ids = aspect_ids.squeeze()  # (batch, 1) --> (batch, )
        
        # (batch, input_dim, polarity_num)
        W1 = self.W1_embs(aspect_ids).view(batch_size, self.input_dim, self.polarity_num)
        # (batch, polarity_num)
        b1 = self.b1_embs(aspect_ids)
        
        # predict
        logits = torch.bmm(X.unsqueeze(dim=1), W1).squeeze(dim=1)  # (batch, polarity_num)
        logits = torch.add(logits, b1)
        
        return logits
    
    
class End2End_LSTM(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(End2End_LSTM, self).__init__()
        self.opt = opt
        self.embed_dim = opt.embed_dim
        self.aspect_num = 57  # for MASAD dataset
        
        if opt.dataset != 'MASAD' or (opt.dataset == 'MASAD' and opt.shared_emb == 1):
            self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        elif opt.dataset == 'MASAD' and opt.shared_emb == 0:
            word_embedding_matrix, aspect_embedding_matrix = embedding_matrix[0], embedding_matrix[1]
            self.embed = nn.Embedding.from_pretrained(torch.tensor(word_embedding_matrix, dtype=torch.float), freeze=False)
            self.aspect_embed = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float), freeze=False)
        
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        self.aspect_pred_layer = Aspect_Prediction(opt.hidden_dim*2+opt.embed_dim, opt.polarities_dim, self.aspect_num)

    def forward(self, inputs):
        text_raw_indices = inputs[0]  # (batch, common_review_length); 句子不定长，统一[PAD]到相同长度，[PAD]对应id为0
        aspect_indices = inputs[1]  # (batch, common_aspect_length);  Aspect不定长，统一[PAD]到相同长度，[PAD]对应id为0
        aspect_ids = inputs[2]  # (batch, ), refers to aspect for each sample
        
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
        
        # 3. 将review embeddings输入双向LSTM，获得隐向量
        word_hidden_vec, (hn, cn) = self.bi_lstm(x, x_len)
        
        # 4. CONCAT: forward hn -- reverse hn -- aspect emb
        hn_forward = hn[0, :, :]  # (batch, hidden_dim)
        hn_reverse = hn[1, :, :]  # (batch, hidden_dim)
        review_aspect_vec = torch.cat([hn_forward, hn_reverse, aspect], dim=-1)  # (batch, hidden_dim*2+embed_dim)
        
        # 5. 预测层
        out = self.aspect_pred_layer(review_aspect_vec, aspect_ids)
        
        return out
