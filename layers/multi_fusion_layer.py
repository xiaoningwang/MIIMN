import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bilinear_Fusion_Layer(nn.Module):
    def __init__(self, input_dim_X, input_dim_Y, output_dim):
        '''bilinear_fusion_layer
        
        input: 
            1. features X from Modal-X. (batch, input_dim_X)
            2. features Y from Modal-Y. (batch, input_dim_Y)
            
        output:
            features H fused information from X and Y. (batch, output_dim)
            
        pipeline:
            H = X*W*Y + b
            dim(W) = (input_dim_X, output_dim, input_dim_Y)
            dim(b) = (output_dim, )
        '''
        super(Bilinear_Fusion_Layer, self).__init__()
        
        self.input_dim_X = input_dim_X
        self.input_dim_Y = input_dim_Y
        self.output_dim = output_dim if output_dim is not None else 100
        
        self.W = nn.Parameter(torch.FloatTensor(output_dim, input_dim_X, input_dim_Y))
        self.b = nn.Parameter(torch.FloatTensor(output_dim, ))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        bound = 1. / math.sqrt(self.input_dim_X)
        self.W.data.uniform_(-bound, bound)
        self.b.data.uniform_(-bound, bound)
    
    def forward(self, X, Y):
        X = X.unsqueeze(dim=0)  # (batch, input_dim_X) --> (1, batch, input_dim_X)
        X = X.expand(self.output_dim, -1, -1)  # (output_dim, batch, input_dim_X)
        
        XX = torch.bmm(X, self.W)  # (output_dim, batch, input_dim_Y)
        XX = XX.permute(1, 0, 2)  # (batch, output_dim, input_dim_Y)
        
        Y = Y.unsqueeze(dim=-1)  # (batch, input_dim_Y) --> (batch, input_dim_Y, 1)
        
        out = torch.bmm(XX, Y).squeeze()  # (batch, output_dim)
        out = torch.add(out, self.b)
        
        return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_num):
        super(SelfAttention, self).__init__()
        
        self.head_num = head_num
        self.input_dim = input_dim
        self.hidden_dim = input_dim//head_num
        
        self.W_Q = nn.Parameter(torch.FloatTensor(head_num, input_dim, self.hidden_dim))
        self.W_K = nn.Parameter(torch.FloatTensor(head_num, input_dim, self.hidden_dim))
        self.W_V = nn.Parameter(torch.FloatTensor(head_num, input_dim, self.hidden_dim))
        self.proj = nn.Linear(input_dim, input_dim, bias=False)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        bound = 1. / math.sqrt(self.hidden_dim)
        self.W_Q.data.uniform_(-bound, bound)
        self.W_K.data.uniform_(-bound, bound)
        self.W_V.data.uniform_(-bound, bound)
    
    def forward(self, X_Q, X_K, X_V):
        '''
        X_Q.shape = (batch, q_len, input_dim)
        X_K.shape = (batch, k_len, input_dim)
        X_V.shape = (batch, v_len, input_dim)
        '''
        batch_size = X_Q.shape[0]
        q_len, k_len, v_len = X_Q.shape[1], X_K.shape[1], X_V.shape[1]
        
        X_Q = X_Q.repeat(self.head_num, 1, 1)  # (batch*head_num, q_len, input_dim)
        X_Q = X_Q.view(self.head_num, -1, self.input_dim)  #(head_num, batch*q_len, input_dim)
        X_K = X_K.repeat(self.head_num, 1, 1)  # (batch*head_num, k_len, input_dim)
        X_K = X_K.view(self.head_num, -1, self.input_dim)  #(head_num, batch*k_len, input_dim)
        X_V = X_V.repeat(self.head_num, 1, 1)  # (batch*head_num, v_len, input_dim)
        X_V = X_V.view(self.head_num, -1, self.input_dim)  #(head_num, batch*v_len, input_dim)
        
        # Convert X to Q/K/V
        # --> (head_num, batch*input_num, hidden_dim) --> (head_num*batch, input_num, hidden_dim)
        Q = torch.bmm(X_Q, self.W_Q).view(-1, q_len, self.hidden_dim)
        K = torch.bmm(X_K, self.W_K).view(-1, k_len, self.hidden_dim)
        V = torch.bmm(X_V, self.W_V).view(-1, v_len, self.hidden_dim)
        
        # Scaled Dot-Product
        dot_prod = torch.bmm(Q, K.permute(0, 2, 1))  # (head_num*batch, q_len, k_len)
        scaled_dot_prod = torch.div(dot_prod, math.sqrt(self.hidden_dim))
        
        # Softmax
        att_weights = F.softmax(scaled_dot_prod, dim=-1)  # (head_num*batch, q_len, k_len)
        
        # Weighted Sum
        att_values = torch.bmm(att_weights, V)  # (head_num*batch, q_len, hidden_dim)
        
        # Projection
        att_values = torch.cat(torch.split(att_values, batch_size, dim=0), dim=-1)  # (batch, q_len, head_num*hidden_dim)
        multi_head = self.proj(att_values)  # (batch, q_len, input_dim)
        
        return multi_head


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=None):
        super(FFN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim*2
        
        self.layer1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.layer2 = nn.Linear(self.hidden_dim, self.input_dim)
        
    def forward(self, X):
        '''
        X.shape = (batch, input_num, input_dim)
        '''
        batch_size, input_num = X.shape[0], X.shape[1]
        X = X.view(-1, self.input_dim)  # (batch, input_num, input_dim) --> (batch*input_num, input_dim)
        
        X = self.layer1(X)  # (batch*input_num, hidden_dim)
        X = F.relu(X)
        X = self.layer2(X)  # (batch*input_num, input_dim)
        
        X = X.view(batch_size, input_num, self.input_dim)  # (batch, input_num, input_dim)
        
        return X


class Transformer_Fusion_Layer(nn.Module):
    def __init__(self, input_dim, head_num=4):
        super(Transformer_Fusion_Layer, self).__init__()
        '''transformer_fusion_layer(namely transformer encoder)
        
        input:
            features to fuse, X. (batch, input_num, input_dim)
            
        output:
            fused feature H. (batch, input_num, input_dim)
        
        pipeline:
            1. Layer1 : Multi-Head Self-Attention Layer
                head_i = SelfAttention(X * W_Qi, X * W_Ki, X * W_Vi) --- No Bias
                multi_head = CONCAT(head_1, head_2, ..., head_n) * W_proj --- No Bias
                layer1_out = LayerNorm(multi_head + X)
                
            2. Layer2 : Point-wise Feed-Forward Layer
                ffn_out = Relu(W1*layer1_out + b1)
                ffn_out = W2*fnn_out + b2
                layer2_out = LayerNorm(layer1_out + ffn_out)
        '''
        
        self.input_dim = input_dim
        self.head_num = head_num
        
        self.self_attention_layer = SelfAttention(input_dim, head_num)
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_dim)
        
        self.ffn_layer = FFN(input_dim)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_dim)
        
    def forward(self, Q=None, K=None, V=None):
        '''
        Q.shape = (batch, q_len, input_dim)
        K.shape = (batch, k_len, input_dim)
        V.shape = (batch, v_len, input_dim)
        '''
        if K is None and V is None:  # Self-Attention
            X = Q
            multi_head = self.self_attention_layer(X, X, X)
            out1 = self.layer_norm1(torch.add(X, multi_head))
        elif K is not None and V is not None:  # Cross-Attention
            X_Q, X_K, X_V = Q, K, V
            multi_head = self.self_attention_layer(X_Q, X_K, X_V)
            out1 = self.layer_norm1(torch.add(X_Q, multi_head))
        
        ffn_out = self.ffn_layer(out1)
        out2 = self.layer_norm2(torch.add(out1, ffn_out))
        
        return out2
    
    
if __name__ == '__main__':
    #temp = Bilinear_Fusion_Layer(30, 30, 10)
    #X = torch.FloatTensor([i for i in range(120)]).reshape(4,30)
    #Y = torch.FloatTensor([i for i in range(120)]).reshape(4,30)
    #res = temp(X, Y)
    #print(res.shape)
    
    temp = Transformer_Fusion_Layer(input_dim=12)
    X = torch.randn(3, 2, 12)
    res = temp(X)
    print(X)
