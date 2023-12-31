import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.multi_fusion_layer import Transformer_Fusion_Layer

class Early_Filter(nn.Module):
    '''
    early filtering
    
    inputs: text_hidden_vec, imgs_fine_grain
    
    outputs: feature_weights, shape = (batch, feature_weight_num)
    '''
    def __init__(self, input_dim, feature_weight_num):
        super(Early_Filter, self).__init__()
        
        self.crossmodal_fusion_layer_text2img = Transformer_Fusion_Layer(input_dim, head_num=3)
        self.crossmodal_fusion_layer_img2text = Transformer_Fusion_Layer(input_dim, head_num=3)
        
        self.linear = nn.Linear(input_dim*2, feature_weight_num)
    
    def forward(self, text_inputs, img_inputs):
        '''
        text_inputs: (batch, text_len, input_dim)
        img_inputs: (batch, img_num, input_dim)
        '''
        text2img_vec = self.crossmodal_fusion_layer_text2img(text_inputs, img_inputs, img_inputs)
        img2text_vec = self.crossmodal_fusion_layer_img2text(img_inputs, text_inputs, text_inputs)
        
        text2img_vec_pooled = torch.mean(text2img_vec, dim=1)  # (batch, input_dim)
        img2text_vec_pooled = torch.mean(img2text_vec, dim=1)  # (batch, input_dim)
        
        input_vec = torch.cat([text2img_vec_pooled, img2text_vec_pooled], dim=-1)  # (batch, input_dim*2)
        out = F.sigmoid(self.linear(input_vec))  # (batch, feature_weight_num)
        
        return out
        
class Late_Filter(nn.Module):
    '''
    late filtering
    
    inputs: v_X_aspect, v_H_aspect, text2img_vec_pooled, img2text_vec_pooled
    
    outputs: feature_weights, shape = (batch, feature_weight_num)
    '''
    def __init__(self, input_dim, feature_weight_num):
        super(Late_Filter, self).__init__()
        
        self.feat_interaction_layer = Transformer_Fusion_Layer(input_dim, head_num=3)
        
        self.linear1 = nn.Linear(input_dim*4, input_dim)
        self.linear2 = nn.Linear(input_dim, feature_weight_num)
        
    def forward(self, text_vec1, text_vec2, cross_vec1, cross_vec2):
        '''
        input shape = (batch, input_dim)
        '''
        batch_size = text_vec1.shape[0]
        
        # (batch, input_dim) --> (batch, 1, input_dim)
        text_vec1 = text_vec1.unsqueeze(dim=1)
        text_vec2 = text_vec2.unsqueeze(dim=1)
        cross_vec1 = cross_vec1.unsqueeze(dim=1)
        cross_vec2 = cross_vec2.unsqueeze(dim=1)
        
        # (batch, 4, input_dim)
        feats = torch.cat([text_vec1, text_vec2, cross_vec1, cross_vec2], dim=1)
        
        # Self-Attention Interaction
        feats = self.feat_interaction_layer(feats)
        feats = feats.view(batch_size, -1)  # (batch, input_dim*4)
        
        # ~SENet
        out = F.relu(self.linear1(feats))
        out = F.sigmoid(self.linear2(out))
        
        return out
