import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.multi_fusion_layer import Bilinear_Fusion_Layer, Transformer_Fusion_Layer
from layers.feature_filtering import Early_Filter, Late_Filter


class FeatFilterModel(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(FeatFilterModel, self).__init__()
        
        self.opt = opt
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # transform layer for img
        self.transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # self-attention layer for img
        self.img_self_attention_layer = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # cross-modal fusion layer
        self.crossmodal_fusion_layer_text2img = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        self.crossmodal_fusion_layer_img2text = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # prediction layer
        self.linear1 = nn.Linear(opt.hidden_dim*4, opt.hidden_dim)
        self.linear2 = nn.Linear(opt.hidden_dim, 2)
    
    def forward(self, inputs):
        '''
        three elements of inputs: 
            sentence token ids,  
            global img embedding,
            fine-grained img embeddings
        '''

        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        imgs_fine_grain = inputs[2]  # (batch, 9, img_emb_dim)
        imgs_fine_grain = self.transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        imgs_fine_grain = self.img_self_attention_layer(imgs_fine_grain)
        
        text2img_vec = self.crossmodal_fusion_layer_text2img(text_hidden_vec, imgs_fine_grain, imgs_fine_grain)
        img2text_vec = self.crossmodal_fusion_layer_img2text(imgs_fine_grain, text_hidden_vec, text_hidden_vec)
        
        text2img_vec_pooled = torch.mean(text2img_vec, dim=1)
        img2text_vec_pooled = torch.mean(img2text_vec, dim=1)
        
        final_in = torch.cat([text2img_vec_pooled, img2text_vec_pooled], dim=1)  # (batch, hidden_dim*4)
        out = F.relu(self.linear1(final_in))
        logits = self.linear2(out)
        
        return logits
        