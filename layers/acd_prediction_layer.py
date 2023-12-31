import torch
import torch.nn as nn
import torch.nn.functional as F


class Aspect_Category_Prediction(nn.Module):
    def __init__(self, embed_dim, hidden_dim, aspect_num=1):
        '''aspect category prediction layer
        
        different aspect refers to different prediction layer
        '''
        super(Aspect_Category_Prediction, self).__init__()
        
        self.W1 = nn.Parameter(torch.FloatTensor(aspect_num, embed_dim+hidden_dim, hidden_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim))
        
        # We adopt SOFTMAX to do binary classification, so the output dim is 2 instead of 1.
        self.W2 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim, 2))
        self.b2 = nn.Parameter(torch.FloatTensor(aspect_num, 2))
        
    def forward(self, v_X, v_H):
        '''
        v_X: (batch, aspect_num, embed_dim)
        v_H: (batch, asepct_num, hidden_dim)
        '''
        batch_size = v_X.shape[0]
        
        V = torch.cat([v_X, v_H], dim = -1)  # (batch, aspect_num, embed_dim+hidden_dim)
        V = V.permute(1, 0, 2)  # (aspect_num, batch, embed_dim+hidden_dim)
        
        # layer1
        out1 = torch.bmm(V, self.W1)  # (aspect_num, batch, hidden_dim)
        out1 = torch.add(out1, self.b1.unsqueeze(dim=1).expand(-1, batch_size, -1))
        out1 = F.relu(out1)
        
        # layer2
        logits = torch.bmm(out1, self.W2)  # (aspect, batch, 2)
        logits = torch.add(logits, self.b2.unsqueeze(dim=1).expand(-1, batch_size, -1))
        logits = logits.permute(1, 0, 2)  # (batch, aspect, 2)
        
        return logits


class Aspect_Category_Prediction_v2(nn.Module):
    def __init__(self, embed_dim, hidden_dim, aspect_num):
        '''
        version2 is designed for pair sample.
        '''
        super(Aspect_Category_Prediction_v2, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.aspect_num = aspect_num
        
        #self.W1 = nn.Parameter(torch.FloatTensor(aspect_num, embed_dim+hidden_dim, hidden_dim))
        #self.b1 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim))
        self.W1_embs = nn.Embedding(aspect_num, (embed_dim+hidden_dim)*hidden_dim)
        self.b1_embs = nn.Embedding(aspect_num, hidden_dim)
        
        # We adopt SOFTMAX to do binary classification, so the output dim is 2 instead of 1.
        #self.W2 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim, 2))
        #self.b2 = nn.Parameter(torch.FloatTensor(aspect_num, 2))
        self.W2_embs = nn.Embedding(aspect_num, hidden_dim*2)
        self.b2_embs = nn.Embedding(aspect_num, 2)
        
    def forward(self, v_X, v_H, aspect_ids):
        '''
        v_X: (batch, embed_dim)
        v_H: (batch, hidden_dim)
        aspect_ids：(batch, ) ----- torch.LongTensor()
        '''
        batch_size = v_X.shape[0]
        if len(aspect_ids.shape) == 2:
            aspect_ids = aspect_ids.squeeze()  # (batch, 1) --> (batch, )
        
        # (batch, embed_dim+hidden_dim)
        V = torch.cat([v_X, v_H], dim = -1)
        # (batch, embed_dim+hidden_dim, hidden_dim)
        W1 = self.W1_embs(aspect_ids).view(batch_size, self.embed_dim+self.hidden_dim, self.hidden_dim)
        # (batch, hidden_dim)
        b1 = self.b1_embs(aspect_ids)
        # (batch, hidden_dim, 2)
        W2 = self.W2_embs(aspect_ids).view(batch_size, self.hidden_dim, 2)
        # (batch, 2)
        b2 = self.b2_embs(aspect_ids)
        
        # layer1
        out1 = torch.bmm(V.unsqueeze(dim=1), W1).squeeze(dim=1)  # (aspect_num, hidden_dim)
        out1 = torch.add(out1, b1)
        out1 = F.relu(out1)
        
        # layer2
        logits = torch.bmm(out1.unsqueeze(dim=1), W2).squeeze(dim=1)  # (batch, 2)
        logits = torch.add(logits, b2)
        
        return logits
        

class Aspect_Category_Prediction_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim, aspect_num):
        '''
        Based on version2.
        
        version3 is designed to used with Transformer_Fusion_Layer.
        '''
        super(Aspect_Category_Prediction_v3, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.aspect_num = aspect_num
        
        #self.W1 = nn.Parameter(torch.FloatTensor(aspect_num, input_dim, hidden_dim))
        #self.b1 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim))
        self.W1_embs = nn.Embedding(aspect_num, input_dim*hidden_dim)
        self.b1_embs = nn.Embedding(aspect_num, hidden_dim)
        
        # We adopt SOFTMAX to do binary classification, so the output dim is 2 instead of 1.
        #self.W2 = nn.Parameter(torch.FloatTensor(aspect_num, hidden_dim, 2))
        #self.b2 = nn.Parameter(torch.FloatTensor(aspect_num, 2))
        self.W2_embs = nn.Embedding(aspect_num, hidden_dim*2)
        self.b2_embs = nn.Embedding(aspect_num, 2)
        
    def forward(self, X, aspect_ids):
        '''
        X: (batch, input_dim)
        aspect_ids：(batch, ) ----- torch.LongTensor()
        '''
        batch_size = X.shape[0]
        if len(aspect_ids.shape) == 2:
            aspect_ids = aspect_ids.squeeze()  # (batch, 1) --> (batch, )
        
        # (batch, input_dim, hidden_dim)
        W1 = self.W1_embs(aspect_ids).view(batch_size, self.input_dim, self.hidden_dim)
        # (batch, hidden_dim)
        b1 = self.b1_embs(aspect_ids)
        # (batch, hidden_dim, 2)
        W2 = self.W2_embs(aspect_ids).view(batch_size, self.hidden_dim, 2)
        # (batch, 2)
        b2 = self.b2_embs(aspect_ids)
        
        # layer1
        out1 = torch.bmm(X.unsqueeze(dim=1), W1).squeeze(dim=1)  # (batch, hidden_dim)
        out1 = torch.add(out1, b1)
        out1 = F.relu(out1)
        
        # layer2
        logits = torch.bmm(out1.unsqueeze(dim=1), W2).squeeze(dim=1)  # (batch, 2)
        logits = torch.add(logits, b2)
        
        return logits


class Aspect_Category_Prediction_v4(nn.Module):
    def __init__(self, input_dim, aspect_num):
        '''
        Based on version2.
        
        version4 is designed to used with Auxiliary Task.
        '''
        super(Aspect_Category_Prediction_v4, self).__init__()
        
        self.input_dim = input_dim
        self.aspect_num = aspect_num
        
        self.W_embs = nn.Embedding(aspect_num, input_dim*2)
        self.b_embs = nn.Embedding(aspect_num, 2)
        
    def forward(self, X, aspect_ids):
        '''
        X: (batch, input_dim)
        aspect_ids：(batch, ) ----- torch.LongTensor()
        '''
        batch_size = X.shape[0]
        if len(aspect_ids.shape) == 2:
            aspect_ids = aspect_ids.squeeze()  # (batch, 1) --> (batch, )
        
        # (batch, input_dim, 2)
        W = self.W_embs(aspect_ids).view(batch_size, self.input_dim, 2)
        # (batch, 2)
        b = self.b_embs(aspect_ids)
        
        logits = torch.bmm(X.unsqueeze(dim=1), W).squeeze(dim=1)  # (batch, 2)
        logits = torch.add(logits, b)
        
        return logits
    
    
if __name__ == '__main__':
    input1 = torch.FloatTensor(3,4,7)
    input2 = torch.FloatTensor(3,4,5)
    
    acd_prediction_layer = Aspect_Category_Prediction(7, 5, 4)
    res = acd_prediction_layer(input1, input2)
        