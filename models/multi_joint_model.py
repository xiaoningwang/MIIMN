import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.aspect_attention import Aspect_Attention, Aspect_Attention_v2
from layers.sentiment_attention import Sentiment_Attention, Sentiment_Attention_v2
from layers.acd_prediction_layer import Aspect_Category_Prediction, Aspect_Category_Prediction_v2, Aspect_Category_Prediction_v3, Aspect_Category_Prediction_v4
from layers.asc_prediction_layer import Shared_Sentiment_Prediction, Shared_Sentiment_Prediction_v2, Shared_Sentiment_Prediction_v3
from layers.multi_fusion_layer import Bilinear_Fusion_Layer, Transformer_Fusion_Layer
from layers.feature_filtering import Early_Filter, Late_Filter

from models.feat_filter_model import FeatFilterModel




"""
Multi-Joint-Model
"""
class MultiJointModel_complete_v2(nn.Module):
    '''
    MultiJointModel_complete_v2 = MultiJointModel_temp8(ACD改进) + MultiJointModel_temp9_modified_v2(ASC改进)
    
    involve fine-grained(local-level) image features for ACD task.
    
    involve global-level and scene-level image features for ASC task.
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_complete_v2, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD Network '''
        # aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_text2img = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_img2text = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # transform layer for img
        self.acd_transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # self-attention layer for img
        self.acd_img_self_attention_layer = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # cross-modal fusion layer
        self.acd_crossmodal_fusion_layer_text2img = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        self.acd_crossmodal_fusion_layer_img2text = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*8, opt.hidden_dim, self.aspect_num)
        
        
        ''' ASC Network '''        
        # the number of multi-modal interaction block
        self.interact_block_num = 3
        # 1 for text_hidden_vec, 2 for text_raw_emb
        self.attention_text2text_layer1_list = []
        self.attention_text2text_layer2_list = []
        self.attention_global_img2text_layer1_list = []
        self.attention_global_img2text_layer2_list = []
        self.attention_scene_img2text_layer1_list = []
        self.attention_scene_img2text_layer2_list = []
        self.self_attention_layer_list = []
        # multi-modal interactrion block
        for i in range(self.interact_block_num):
            self.attention_text2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_text2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.self_attention_layer_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
        
        # image transform layer
        self.asc_transform_img_global = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for global-level img feat
        self.asc_transfrom_img_scene = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for scene-level img feat
        
        # shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.hidden_dim*2, opt.hidden_dim*4, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        seven elements of inputs: 
            sentence token ids, 
            aspect ids, 
            global-level image feature,
            img num,
            fine-grained image feature,
            text_raw_indices_trc,
            scene-level image feature
        '''
        
        ''' input processing '''
        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        #text_raw_emb = self.dropout(text_raw_emb)
        #text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        imgs = inputs[2]  # (batch, max_img_num, img_emb_dim)
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        global_img_feat = F.tanh(self.asc_transform_img_global(imgs)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        imgs_scene_level = inputs[6]
        if self.opt.dataset == 'MASAD' and len(imgs_scene_level.shape) == 2:
            imgs_scene_level = imgs_scene_level.unsqueeze(dim=1)
        scene_img_feat = F.tanh(self.asc_transfrom_img_scene(imgs_scene_level)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        imgs_fine_grain = inputs[4]  # (batch, 9, img_emb_dim)
        imgs_fine_grain = self.acd_transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        
        
        ''' ACD Network '''
        # generate aspect-specific text representation
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # self attention interaction for fine-grained imgs
        imgs_fine_grain = self.acd_img_self_attention_layer(imgs_fine_grain)  # (batch, 9, hidden_dim*2)
        
        # text-to-img cross-modal interaction
        text2img_vec = self.acd_crossmodal_fusion_layer_text2img(text_hidden_vec, imgs_fine_grain, imgs_fine_grain)  # (batch, max_seq_len, hidden_dim*2)
        img2text_vec = self.acd_crossmodal_fusion_layer_img2text(imgs_fine_grain, text_hidden_vec, text_hidden_vec)  # (batch, 9, hidden_dim*2)
        
        # aspect-specific visual attention
        v_text2img_aspect = self.aspect_attention_text2img(text2img_vec, aspect_ids)  # (batch, hidden_dim*2)
        v_img2text_aspect = self.aspect_attention_img2text(img2text_vec, aspect_ids)  # (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_inputs = torch.cat([v_X_aspect.unsqueeze(dim=1), v_H_aspect.unsqueeze(dim=1), v_text2img_aspect.unsqueeze(dim=1), v_img2text_aspect.unsqueeze(dim=1)], dim=-1)  # (batch, 1, hidden_dim*8)
        acd_inputs = acd_inputs.squeeze(dim=1)
        acd_logits = self.acd_pred_layer(acd_inputs, aspect_ids)
        
        
        ''' ASC Network '''  
        deep_text_feat1, deep_text_feat2 = v_H_aspect, v_X_aspect
        deep_global_img_feat1, deep_global_img_feat2 = global_img_feat, global_img_feat
        deep_scene_img_feat1, deep_scene_img_feat2 = scene_img_feat, scene_img_feat
        
        for i in range(self.interact_block_num):
            # way of text_hidden_vec
            att_feat_text2text1 = self.attention_text2text_layer1_list[i](deep_text_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_global_img2text1 = self.attention_global_img2text_layer1_list[i](deep_global_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_scene_img2text1 = self.attention_scene_img2text_layer1_list[i](deep_scene_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            # way of text_raw_emb
            att_feat_text2text2 = self.attention_text2text_layer2_list[i](deep_text_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_global_img2text2 = self.attention_global_img2text_layer2_list[i](deep_global_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_scene_img2text2 = self.attention_scene_img2text_layer2_list[i](deep_scene_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            # self-attention interaction
            feat_to_interact = torch.cat([att_feat_text2text1, att_feat_global_img2text1, att_feat_scene_img2text1, att_feat_text2text2, att_feat_global_img2text2, att_feat_scene_img2text2], dim=1)  # (batch, 6, hidden_dim*2)
            feat_interacted = self.self_attention_layer_list[i](feat_to_interact)  # (batch, 6, hidden_dim*2)
            # split to six feature vectors
            deep_text_feat1, deep_global_img_feat1, deep_scene_img_feat1, deep_text_feat2, deep_global_img_feat2, deep_scene_img_feat2 = torch.split(feat_interacted, 1, dim=1)
            # (batch, 1, hidden_dim*2) --> (batch, hidden_dim*2)
            deep_text_feat1, deep_text_feat2 = deep_text_feat1.squeeze(dim=1), deep_text_feat2.squeeze(dim=1)
            deep_global_img_feat1, deep_global_img_feat2 = deep_global_img_feat1.squeeze(dim=1), deep_global_img_feat2.squeeze(dim=1)
            deep_scene_img_feat1, deep_scene_img_feat2 = deep_scene_img_feat1.squeeze(dim=1), deep_scene_img_feat2.squeeze(dim=1)

        deep_text_feat = (deep_text_feat1 + deep_text_feat2) / 2
        deep_global_img_feat = (deep_global_img_feat1 + deep_global_img_feat2) / 2
        deep_scene_img_feat = (deep_scene_img_feat1 + deep_scene_img_feat2) / 2
        deep_img_feat = torch.cat([deep_global_img_feat, deep_scene_img_feat], dim=1)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(deep_text_feat, deep_img_feat)
        
        return acd_logits, asc_logits




"""
ACD Ablation Model
"""
class MultiJointModel_temp8(nn.Module):
    '''
    以A-Joint-Model为基础，保持ASC子网络结构不变，在ACD子网络中引入“细粒度”图像信息。
    
    temp8: 
    1. first conduct Text-to-Image Multi-modal Fusion based on <Cross-Modal Multi-Head Attention>
            with text_hidden_vec as Q
            with fine-grained image features as K and V
    2. extract <Aspect-Specific Visual Representations> v_Img_aspect by Aspect-Specific Visual Attention
            
    --> 新增一个线性层，负责变换图像特征维度
    --> 新增一个Multi-Head Self-Attention层，负责图像模态内的信息交互
    --> 新增一个Cross-Modal Multi-Head Attention层，负责跨模态的信息交互
    --> 新增一个Aspect-Specific Visual Attention层，负责产生v_Img_aspect
    --> 采用Aspect_Category_Prediction_v3
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp8, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD Network '''
        # aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_text2img = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_img2text = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # transform layer for img
        self.transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # self-attention layer for img
        self.img_self_attention_layer = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # cross-modal fusion layer
        self.crossmodal_fusion_layer_text2img = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        self.crossmodal_fusion_layer_img2text = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*8, opt.hidden_dim, self.aspect_num)
        
        
        ''' ASC Network '''
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        five elements of inputs: 
            sentence token ids, 
            aspect ids, 
            global img embedding,
            img num,
            fine-grained img embeddings
        '''
        ''' input processing '''
        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        #text_raw_emb = self.dropout(text_raw_emb)
        #text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        imgs_fine_grain = inputs[4]  # (batch, 9, img_emb_dim)
        imgs_fine_grain = self.transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        
        ''' ACD Network '''
        # generate aspect-specific text representation
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # self attention interaction for fine-grained imgs
        imgs_fine_grain = self.img_self_attention_layer(imgs_fine_grain)  # (batch, 9, hidden_dim*2)
        
        # text-to-img cross-modal interaction
        text2img_vec = self.crossmodal_fusion_layer_text2img(text_hidden_vec, imgs_fine_grain, imgs_fine_grain)  # (batch, max_seq_len, hidden_dim*2)
        img2text_vec = self.crossmodal_fusion_layer_img2text(imgs_fine_grain, text_hidden_vec, text_hidden_vec)  # (batch, 9, hidden_dim*2)
        
        # aspect-specific visual attention
        v_text2img_aspect = self.aspect_attention_text2img(text2img_vec, aspect_ids)  # (batch, hidden_dim*2)
        v_img2text_aspect = self.aspect_attention_img2text(img2text_vec, aspect_ids)  # (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_inputs = torch.cat([v_X_aspect.unsqueeze(dim=1), v_H_aspect.unsqueeze(dim=1), v_text2img_aspect.unsqueeze(dim=1), v_img2text_aspect.unsqueeze(dim=1)], dim=-1)  # (batch, 1, hidden_dim*8)
        acd_inputs = acd_inputs.squeeze(dim=1)
        acd_logits = self.acd_pred_layer(acd_inputs, aspect_ids)
        
        
        ''' ASC Network '''
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits


class MultiJointModel_temp8_GlobalImgFeat(nn.Module):
    '''
    将MultiJointModel_temp8使用的fine-grained image feat替换为global-level image feat，保持模型结构不变。
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp8_GlobalImgFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD Network '''
        # aspect attention layer
        # H is the output of bi_lstm layer, is's bi-directional, so the size of H is 2*hidden_dim
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_text2img = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_img2text = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # transform layer for img
        self.transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # self-attention layer for img
        self.img_self_attention_layer = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # cross-modal fusion layer
        self.crossmodal_fusion_layer_text2img = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        self.crossmodal_fusion_layer_img2text = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=6)
        
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*8, opt.hidden_dim, self.aspect_num)
        
        
        ''' ASC Network '''
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        five elements of inputs: 
            sentence token ids, 
            aspect ids, 
            global img embedding,
            img num,
            fine-grained img embeddings
        '''
        ''' input processing '''
        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        #text_raw_emb = self.dropout(text_raw_emb)
        #text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        #imgs_fine_grain = inputs[4]  # (batch, 9, img_emb_dim)
        #imgs_fine_grain = self.transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        
        imgs = inputs[2]  # coarse-grained img feat
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        imgs_fine_grain = self.transform_img(imgs)  # is actually coarse-grained img feat
        #imgs_fine_grain = imgs_fine_grain.expand(-1, 9, -1)  # (batch, 1, hidden_dim*2) --> (batch, 9, hidden_dim*2)


        ''' ACD Network '''
        # generate aspect-specific text representation
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # self attention interaction for fine-grained imgs
        imgs_fine_grain = self.img_self_attention_layer(imgs_fine_grain)  # (batch, 9, hidden_dim*2)
        
        # text-to-img cross-modal interaction
        text2img_vec = self.crossmodal_fusion_layer_text2img(text_hidden_vec, imgs_fine_grain, imgs_fine_grain)  # (batch, max_seq_len, hidden_dim*2)
        img2text_vec = self.crossmodal_fusion_layer_img2text(imgs_fine_grain, text_hidden_vec, text_hidden_vec)  # (batch, 9, hidden_dim*2)
        
        # aspect-specific visual attention
        v_text2img_aspect = self.aspect_attention_text2img(text2img_vec, aspect_ids)  # (batch, hidden_dim*2)
        v_img2text_aspect = self.aspect_attention_img2text(img2text_vec, aspect_ids)  # (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_inputs = torch.cat([v_X_aspect.unsqueeze(dim=1), v_H_aspect.unsqueeze(dim=1), v_text2img_aspect.unsqueeze(dim=1), v_img2text_aspect.unsqueeze(dim=1)], dim=-1)  # (batch, 1, hidden_dim*8)
        acd_inputs = acd_inputs.squeeze(dim=1)
        acd_logits = self.acd_pred_layer(acd_inputs, aspect_ids)
        
        
        ''' ASC Network '''
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits


class MultiJointModel_ACD_Ablation_GlobalImgFeat(nn.Module):
    '''
    在A-Joint-Model的基础上，为ACD子网络引入global-level image feature，与文本表征拼接使用。
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_ACD_Ablation_GlobalImgFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD Network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # transform layer for img
        self.transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*6, opt.hidden_dim, self.aspect_num)
        
        
        ''' ASC Network '''
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        five elements of inputs: 
            sentence token ids, 
            aspect ids, 
            global img embedding,
            img num,
            fine-grained img embeddings
        '''
        ''' input processing '''
        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        #text_raw_emb = self.dropout(text_raw_emb)
        #text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        #imgs_fine_grain = inputs[4]  # (batch, 9, img_emb_dim)
        #imgs_fine_grain = self.transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        
        imgs = inputs[2]  # coarse-grained img feat
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        img_hidden_vec = self.transform_img(imgs)  # coarse-grained img feat


        ''' ACD Network '''
        # generate aspect-specific text representation
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_inputs = torch.cat([v_X_aspect.unsqueeze(dim=1), v_H_aspect.unsqueeze(dim=1), img_hidden_vec], dim=-1)  # (batch, 1, hidden_dim*6)
        acd_inputs = acd_inputs.squeeze(dim=1)
        acd_logits = self.acd_pred_layer(acd_inputs, aspect_ids)
        
        
        ''' ASC Network '''
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits


class MultiJointModel_ACD_Ablation_LocalImgFeat(nn.Module):
    '''
    在A-Joint-Model的基础上，为ACD子网络引入local-level image features，
    经过aspect-specific attention层缩减成单个图像特征后，与文本表征拼接使用。
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_ACD_Ablation_LocalImgFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD Network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_img = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        
        # transform layer for img
        self.transform_img = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)
        
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v3(opt.hidden_dim*6, opt.hidden_dim, self.aspect_num)
        
        
        ''' ASC Network '''
        # 5. sentiment attention layer
        self.sentiment_attention = Sentiment_Attention_v2()
        
        # 6. shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        five elements of inputs: 
            sentence token ids, 
            aspect ids, 
            global img embedding,
            img num,
            fine-grained img embeddings
        '''
        ''' input processing '''
        batch_size = inputs[0].shape[0]
        
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        #text_raw_emb = self.dropout(text_raw_emb)
        #text_hidden_vec = self.dropout(text_hidden_vec)
        
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        imgs_fine_grain = inputs[4]  # (batch, 9, img_emb_dim)
        imgs_fine_grain = self.transform_img(imgs_fine_grain.view(-1, self.opt.embed_dim_img)).view(batch_size, 9, self.opt.hidden_dim*2)
        
        #imgs = inputs[2]  # coarse-grained img feat
        #if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
        #    imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        #img_hidden_vec = self.transform_img(imgs)  # coarse-grained img feat


        ''' ACD Network '''
        # generate aspect-specific text representation
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        img_hidden_vec = self.aspect_attention_img(imgs_fine_grain, aspect_ids)  # (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_inputs = torch.cat([v_X_aspect.unsqueeze(dim=1), v_H_aspect.unsqueeze(dim=1), img_hidden_vec.unsqueeze(dim=1)], dim=-1)  # (batch, 1, hidden_dim*6)
        acd_inputs = acd_inputs.squeeze(dim=1)
        acd_logits = self.acd_pred_layer(acd_inputs, aspect_ids)
        
        
        ''' ASC Network '''
        v_X_sentiment = self.sentiment_attention(text_raw_emb, v_X_aspect)  # v_X_Sj, (batch, embed_dim)
        v_H_sentiment = self.sentiment_attention(text_hidden_vec, v_H_aspect)  # v_H_Sj, (batch, hidden_dim*2)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(v_X_sentiment, v_H_sentiment)
        
        return acd_logits, asc_logits
        # return asc_logits




"""
ASC Ablation Model
"""
class MultiJointModel_temp9_modified_v2(nn.Module):
    '''
    involve both global-level image feauture and local-level image features
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp9_modified_v2, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm_text = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        
        
        ''' ASC network '''
        # the number of multi-modal interaction block
        self.interact_block_num = 3
        # 1 for text_hidden_vec, 2 for text_raw_emb
        self.attention_text2text_layer1_list = []
        self.attention_text2text_layer2_list = []
        self.attention_global_img2text_layer1_list = []
        self.attention_global_img2text_layer2_list = []
        self.attention_scene_img2text_layer1_list = []
        self.attention_scene_img2text_layer2_list = []
        self.self_attention_layer_list = []
        # multi-modal interactrion block
        for i in range(self.interact_block_num):
            self.attention_text2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_text2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.self_attention_layer_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
        
        # image transform layer
        self.transform_img_global = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for global-level img feat
        self.transfrom_img_scene = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for scene-level img feat
        
        # shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.hidden_dim*2, opt.hidden_dim*4, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        seven elements of inputs:
        
        'text_raw_indices', 
        'aspect_id', 
        'imgs', 
        'num_imgs', 
        'imgs_fine_grain', 
        'text_raw_indices_trc'
        'imgs_scene_level'
        
        '''
        ''' input processing '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        imgs = inputs[2]  # (batch, max_img_num, img_emb_dim)
        
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm_text(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        imgs_scene_level = inputs[6]
        if self.opt.dataset == 'MASAD' and len(imgs_scene_level.shape) == 2:
            imgs_scene_level = imgs_scene_level.unsqueeze(dim=1)
            
        global_img_feat = F.tanh(self.transform_img_global(imgs)).squeeze(dim=1)  # (batch, hidden_dim*2)
        scene_img_feat = F.tanh(self.transfrom_img_scene(imgs_scene_level)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        
        ''' ACD network '''
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect, aspect_ids)
        
        
        ''' ASC network '''
        deep_text_feat1, deep_text_feat2 = v_H_aspect, v_X_aspect
        deep_global_img_feat1, deep_global_img_feat2 = global_img_feat, global_img_feat
        deep_scene_img_feat1, deep_scene_img_feat2 = scene_img_feat, scene_img_feat
        
        for i in range(self.interact_block_num):
            # way of text_hidden_vec
            att_feat_text2text1 = self.attention_text2text_layer1_list[i](deep_text_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_global_img2text1 = self.attention_global_img2text_layer1_list[i](deep_global_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_scene_img2text1 = self.attention_scene_img2text_layer1_list[i](deep_scene_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            # way of text_raw_emb
            att_feat_text2text2 = self.attention_text2text_layer2_list[i](deep_text_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_global_img2text2 = self.attention_global_img2text_layer2_list[i](deep_global_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_scene_img2text2 = self.attention_scene_img2text_layer2_list[i](deep_scene_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            # self-attention interaction
            feat_to_interact = torch.cat([att_feat_text2text1, att_feat_global_img2text1, att_feat_scene_img2text1, att_feat_text2text2, att_feat_global_img2text2, att_feat_scene_img2text2], dim=1)  # (batch, 6, hidden_dim*2)
            feat_interacted = self.self_attention_layer_list[i](feat_to_interact)  # (batch, 6, hidden_dim*2)
            # split to six feature vectors
            deep_text_feat1, deep_global_img_feat1, deep_scene_img_feat1, deep_text_feat2, deep_global_img_feat2, deep_scene_img_feat2 = torch.split(feat_interacted, 1, dim=1)
            # (batch, 1, hidden_dim*2) --> (batch, hidden_dim*2)
            deep_text_feat1, deep_text_feat2 = deep_text_feat1.squeeze(dim=1), deep_text_feat2.squeeze(dim=1)
            deep_global_img_feat1, deep_global_img_feat2 = deep_global_img_feat1.squeeze(dim=1), deep_global_img_feat2.squeeze(dim=1)
            deep_scene_img_feat1, deep_scene_img_feat2 = deep_scene_img_feat1.squeeze(dim=1), deep_scene_img_feat2.squeeze(dim=1)
        
        deep_text_feat = (deep_text_feat1 + deep_text_feat2) / 2
        deep_global_img_feat = (deep_global_img_feat1 + deep_global_img_feat2) / 2
        deep_scene_img_feat = (deep_scene_img_feat1 + deep_scene_img_feat2) / 2
        deep_img_feat = torch.cat([deep_global_img_feat, deep_scene_img_feat], dim=1)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(deep_text_feat, deep_img_feat)
        
        return acd_logits, asc_logits


class MultiJointModel_temp9_modified_v2_GlobalImgFeat(nn.Module):
    '''
    based on MultiJointModel_temp9_modified_v2, only involve global-level image feauture
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp9_modified_v2_GlobalImgFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm_text = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        
        
        ''' ASC network '''
        # the number of multi-modal interaction block
        self.interact_block_num = 3
        # 1 for text_hidden_vec, 2 for text_raw_emb
        self.attention_text2text_layer1_list = []
        self.attention_text2text_layer2_list = []
        self.attention_global_img2text_layer1_list = []
        self.attention_global_img2text_layer2_list = []
        #self.attention_scene_img2text_layer1_list = []
        #self.attention_scene_img2text_layer2_list = []
        self.self_attention_layer_list = []
        # multi-modal interactrion block
        for i in range(self.interact_block_num):
            self.attention_text2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_text2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_global_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            #self.attention_scene_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            #self.attention_scene_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.self_attention_layer_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
        
        # image transform layer
        self.transform_img_global = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for global-level img feat
        #self.transfrom_img_scene = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for scene-level img feat
        
        # shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.hidden_dim*2, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        seven elements of inputs:
        
        'text_raw_indices', 
        'aspect_id', 
        'imgs', 
        'num_imgs', 
        'imgs_fine_grain', 
        'text_raw_indices_trc'
        'imgs_scene_level'
        
        '''
        ''' input processing '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        imgs = inputs[2]  # (batch, max_img_num, img_emb_dim)
        
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm_text(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
            imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        #imgs_scene_level = inputs[6]
        #if self.opt.dataset == 'MASAD' and len(imgs_scene_level.shape) == 2:
        #    imgs_scene_level = imgs_scene_level.unsqueeze(dim=1)
            
        global_img_feat = F.tanh(self.transform_img_global(imgs)).squeeze(dim=1)  # (batch, hidden_dim*2)
        #scene_img_feat = F.tanh(self.transfrom_img_scene(imgs_scene_level)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        
        ''' ACD network '''
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect, aspect_ids)
        
        
        ''' ASC network '''
        deep_text_feat1, deep_text_feat2 = v_H_aspect, v_X_aspect
        deep_global_img_feat1, deep_global_img_feat2 = global_img_feat, global_img_feat
        #deep_scene_img_feat1, deep_scene_img_feat2 = scene_img_feat, scene_img_feat
        
        for i in range(self.interact_block_num):
            # way of text_hidden_vec
            att_feat_text2text1 = self.attention_text2text_layer1_list[i](deep_text_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_global_img2text1 = self.attention_global_img2text_layer1_list[i](deep_global_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            #att_feat_scene_img2text1 = self.attention_scene_img2text_layer1_list[i](deep_scene_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            
            # way of text_raw_emb
            att_feat_text2text2 = self.attention_text2text_layer2_list[i](deep_text_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_global_img2text2 = self.attention_global_img2text_layer2_list[i](deep_global_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            #att_feat_scene_img2text2 = self.attention_scene_img2text_layer2_list[i](deep_scene_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            
            # self-attention interaction
            feat_to_interact = torch.cat([att_feat_text2text1, att_feat_global_img2text1, att_feat_text2text2, att_feat_global_img2text2], dim=1)  # (batch, 4, hidden_dim*2)
            feat_interacted = self.self_attention_layer_list[i](feat_to_interact)  # (batch, 4, hidden_dim*2)
            # split to four feature vectors
            deep_text_feat1, deep_global_img_feat1, deep_text_feat2, deep_global_img_feat2 = torch.split(feat_interacted, 1, dim=1)
            # (batch, 1, hidden_dim*2) --> (batch, hidden_dim*2)
            deep_text_feat1, deep_text_feat2 = deep_text_feat1.squeeze(dim=1), deep_text_feat2.squeeze(dim=1)
            deep_global_img_feat1, deep_global_img_feat2 = deep_global_img_feat1.squeeze(dim=1), deep_global_img_feat2.squeeze(dim=1)
        
        deep_text_feat = (deep_text_feat1 + deep_text_feat2) / 2
        deep_global_img_feat = (deep_global_img_feat1 + deep_global_img_feat2) / 2
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(deep_text_feat, deep_global_img_feat)
        
        return acd_logits, asc_logits


class MultiJointModel_temp9_modified_v2_SceneImgFeat(nn.Module):
    '''
    based on MultiJointModel_temp9_modified_v2, only involve scene-level image feauture
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp9_modified_v2_SceneImgFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm_text = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        
        
        ''' ASC network '''
        # the number of multi-modal interaction block
        self.interact_block_num = 3
        # 1 for text_hidden_vec, 2 for text_raw_emb
        self.attention_text2text_layer1_list = []
        self.attention_text2text_layer2_list = []
        #self.attention_global_img2text_layer1_list = []
        #self.attention_global_img2text_layer2_list = []
        self.attention_scene_img2text_layer1_list = []
        self.attention_scene_img2text_layer2_list = []
        self.self_attention_layer_list = []
        # multi-modal interactrion block
        for i in range(self.interact_block_num):
            self.attention_text2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_text2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            #self.attention_global_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            #self.attention_global_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer1_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.attention_scene_img2text_layer2_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
            self.self_attention_layer_list.append(Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device))
        
        # image transform layer
        #self.transform_img_global = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for global-level img feat
        self.transfrom_img_scene = nn.Linear(opt.embed_dim_img, opt.hidden_dim*2)  # for scene-level img feat
        
        # shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.hidden_dim*2, opt.hidden_dim*2, opt.polarities_dim)
        
        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        seven elements of inputs:
        
        'text_raw_indices', 
        'aspect_id', 
        'imgs', 
        'num_imgs', 
        'imgs_fine_grain', 
        'text_raw_indices_trc'
        'imgs_scene_level'
        
        '''
        ''' input processing '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        #imgs = inputs[2]  # (batch, max_img_num, img_emb_dim)
        
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm_text(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        #if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
        #    imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        
        imgs_scene_level = inputs[6]
        if self.opt.dataset == 'MASAD' and len(imgs_scene_level.shape) == 2:
            imgs_scene_level = imgs_scene_level.unsqueeze(dim=1)
            
        #global_img_feat = F.tanh(self.transform_img_global(imgs)).squeeze(dim=1)  # (batch, hidden_dim*2)
        scene_img_feat = F.tanh(self.transfrom_img_scene(imgs_scene_level)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        
        ''' ACD network '''
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect, aspect_ids)
        
        
        ''' ASC network '''
        deep_text_feat1, deep_text_feat2 = v_H_aspect, v_X_aspect
        #deep_global_img_feat1, deep_global_img_feat2 = global_img_feat, global_img_feat
        deep_scene_img_feat1, deep_scene_img_feat2 = scene_img_feat, scene_img_feat
        
        for i in range(self.interact_block_num):
            # way of text_hidden_vec
            att_feat_text2text1 = self.attention_text2text_layer1_list[i](deep_text_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            #att_feat_global_img2text1 = self.attention_global_img2text_layer1_list[i](deep_global_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            att_feat_scene_img2text1 = self.attention_scene_img2text_layer1_list[i](deep_scene_img_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
            
            # way of text_raw_emb
            att_feat_text2text2 = self.attention_text2text_layer2_list[i](deep_text_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            #att_feat_global_img2text2 = self.attention_global_img2text_layer2_list[i](deep_global_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            att_feat_scene_img2text2 = self.attention_scene_img2text_layer2_list[i](deep_scene_img_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
            
            # self-attention interaction
            feat_to_interact = torch.cat([att_feat_text2text1, att_feat_scene_img2text1, att_feat_text2text2, att_feat_scene_img2text2], dim=1)  # (batch, 4, hidden_dim*2)
            feat_interacted = self.self_attention_layer_list[i](feat_to_interact)  # (batch, 4, hidden_dim*2)
            # split to four feature vectors
            deep_text_feat1, deep_scene_img_feat1, deep_text_feat2, deep_scene_img_feat2 = torch.split(feat_interacted, 1, dim=1)
            # (batch, 1, hidden_dim*2) --> (batch, hidden_dim*2)
            deep_text_feat1, deep_text_feat2 = deep_text_feat1.squeeze(dim=1), deep_text_feat2.squeeze(dim=1)
            deep_scene_img_feat1, deep_scene_img_feat2 = deep_scene_img_feat1.squeeze(dim=1), deep_scene_img_feat2.squeeze(dim=1)
        
        deep_text_feat = (deep_text_feat1 + deep_text_feat2) / 2
        deep_scene_img_feat = (deep_scene_img_feat1 + deep_scene_img_feat2) / 2
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(deep_text_feat, deep_scene_img_feat)
        
        return acd_logits, asc_logits


class MultiJointModel_temp9_modified_v2_OnlyTextFeat(nn.Module):
    '''
    based on MultiJointModel_temp9_modified_v2, only involve text feauture
    '''
    def __init__(self, embedding_matrix, aspect2idx, opt):
        '''
        aspect2idx is a dictionary, aspect_name as key, aspect_id as value
        
        For MASAD, there are 57 aspects with aspect_id from 1 to 57.
        '''
        super(MultiJointModel_temp9_modified_v2_OnlyTextFeat, self).__init__()
        self.opt = opt
        self.aspect_num = len(aspect2idx)
        
        # embedding layer
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False)
        
        # bi-lstm layer
        self.bi_lstm_text = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        
        ''' ACD network '''
        # aspect attention layer
        self.aspect_attention_X = Aspect_Attention_v2(opt.embed_dim, opt.hidden_dim, self.aspect_num)
        self.aspect_attention_H = Aspect_Attention_v2(opt.hidden_dim*2, opt.hidden_dim, self.aspect_num)
        # aspect category prediction layer
        self.acd_pred_layer = Aspect_Category_Prediction_v2(opt.embed_dim, opt.hidden_dim*2, self.aspect_num)
        
        
        ''' ASC network '''
        # multi-modal interactrion block
        self.attention_text2text_layer1 = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device)
        self.attention_text2text_layer2 = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device)
        self.self_attention_layer = Transformer_Fusion_Layer(opt.hidden_dim*2, head_num=3).to(opt.device)
        
        # shared sentiment prediction layer
        self.asc_pred_layer = Shared_Sentiment_Prediction_v2(opt.hidden_dim*2, opt.hidden_dim*2, opt.polarities_dim)

        # Ex. dropout layer
        self.dropout = nn.Dropout(p=opt.dropout)
        
    def forward(self, inputs):
        '''
        seven elements of inputs:
        
        'text_raw_indices', 
        'aspect_id', 
        'imgs', 
        'num_imgs', 
        'imgs_fine_grain', 
        'text_raw_indices_trc'
        'imgs_scene_level'
        
        '''
        ''' input processing '''
        text_raw_indices = inputs[0]  # (batch, max_seq_len)
        aspect_ids = inputs[1]  # (batch, ), refers to aspect for each sample
        #imgs = inputs[2]  # (batch, max_img_num, img_emb_dim)
        
        text_memory_len = torch.sum(text_raw_indices != 0, dim = -1)  # (batch, ); 计算每个评论的真实长度
        text_raw_emb = self.embed(text_raw_indices)  # word embs, (batch, max_seq_len, embed_dim)      
        text_hidden_vec, (_, _) = self.bi_lstm_text(text_raw_emb, text_memory_len)  # hidden vecs, (batch, max_seq_len, hidden_dim*2)
        
        aspect_ids = torch.LongTensor(aspect_ids.cpu()).to(self.opt.device)  # for emb layer
        
        #if self.opt.dataset == 'MASAD' and len(imgs.shape) == 2:
        #    imgs = imgs.unsqueeze(dim=1)  # (batch, img_emb_dim) --> (batch, 1, img_emb_dim)
        #imgs_scene_level = inputs[6]
        #if self.opt.dataset == 'MASAD' and len(imgs_scene_level.shape) == 2:
        #    imgs_scene_level = imgs_scene_level.unsqueeze(dim=1)
            
        #global_img_feat = F.tanh(self.transform_img_global(imgs)).squeeze(dim=1)  # (batch, hidden_dim*2)
        #scene_img_feat = F.tanh(self.transfrom_img_scene(imgs_scene_level)).squeeze(dim=1)  # (batch, hidden_dim*2)
        
        
        ''' ACD network '''
        v_X_aspect = self.aspect_attention_X(text_raw_emb, aspect_ids)  # v_X_Aj, (batch, embed_dim)
        v_H_aspect = self.aspect_attention_H(text_hidden_vec, aspect_ids)  # v_H_Aj, (batch, hidden_dim*2)
        
        # acd prediction layer, output = (batch, 2)
        acd_logits = self.acd_pred_layer(v_X_aspect, v_H_aspect, aspect_ids)
        
        
        ''' ASC network '''
        deep_text_feat1, deep_text_feat2 = v_H_aspect, v_X_aspect
        
        att_feat_text2text1 = self.attention_text2text_layer1(deep_text_feat1.unsqueeze(dim=1), text_hidden_vec, text_hidden_vec)
        att_feat_text2text2 = self.attention_text2text_layer2(deep_text_feat2.unsqueeze(dim=1), text_raw_emb, text_raw_emb)
        
        feat_to_interact = torch.cat([att_feat_text2text1, att_feat_text2text2], dim=1)  # (batch, 2, hidden_dim*2)
        feat_interacted = self.self_attention_layer(feat_to_interact)  # (batch, 2, hidden_dim*2)

        deep_text_feat1, deep_text_feat2 = torch.split(feat_interacted, 1, dim=1)
        deep_text_feat1, deep_text_feat2 = deep_text_feat1.squeeze(dim=1), deep_text_feat2.squeeze(dim=1)
        
        # shared asc prediction layer, output = (batch, polarity_num)
        asc_logits = self.asc_pred_layer(deep_text_feat1, deep_text_feat2)

        return acd_logits, asc_logits