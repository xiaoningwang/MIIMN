import torch
import torch.nn as nn
import torch.nn.functional as F


class Shared_Sentiment_Prediction(nn.Module):
    def __init__(self, embed_dim, hidden_dim, polarity_num):
        '''shared sentiment prediction layer
        
        all aspects share a sentiment prediction layer
        '''
        super(Shared_Sentiment_Prediction, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.layer1 = nn.Linear(embed_dim+hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, polarity_num)
        
    def forward(self, v_X, v_H):
        '''
        v_X: (batch, aspect_num, embed_dim)
        v_H: (batch, asepct_num, hidden_dim)
        '''
        batch_size = v_X.shape[0]
        aspect_num = v_X.shape[1]
        
        V = torch.cat([v_X, v_H], dim = -1)  # (batch, aspect_num, embed_dim+hidden_dim)
        V = V.view(-1, self.embed_dim+self.hidden_dim)  # (batch*aspect_num, embed_dim+hidden_dim)
        
        # layer1
        out1 = self.layer1(V)  # (batch*aspect_num, hidden_dim)
        out1 = F.relu(out1)
        
        # layer2
        logits = self.layer2(out1)  # (batch*aspect_num, polarity_num)
        logits = logits.view(batch_size, aspect_num, -1)  # (batch, aspect_num, polarity_num)
        
        return logits


class Shared_Sentiment_Prediction_v2(nn.Module):
    def __init__(self, embed_dim, hidden_dim, polarity_num):
        '''
        version2 is designed for pair sample.
        '''
        super(Shared_Sentiment_Prediction_v2, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.layer1 = nn.Linear(embed_dim+hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, polarity_num)
        
    def forward(self, v_X, v_H):
        '''
        v_X: (batch, embed_dim)
        v_H: (batch, hidden_dim)
        '''
        batch_size = v_X.shape[0]
        
        V = torch.cat([v_X, v_H], dim=-1)  # (batch, embed_dim+hidden_dim)
        
        # layer1
        out1 = self.layer1(V)  # (batch, hidden_dim)
        out1 = F.relu(out1)
        
        # layer2
        logits = self.layer2(out1)  # (batch, polarity_num)
        
        return logits
    

class Shared_Sentiment_Prediction_v3(nn.Module):
    def __init__(self, input_dim, hidden_dim, polarity_num):
        '''
        based on version2.
        '''
        super(Shared_Sentiment_Prediction_v3, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, polarity_num)
        
    def forward(self, X):
        '''
        X: (batch, input_dim)
        '''        
        # layer1
        out1 = self.layer1(X)  # (batch, hidden_dim)
        out1 = F.relu(out1)
        
        # layer2
        logits = self.layer2(out1)  # (batch, polarity_num)
        
        return logits

if __name__ == '__main__':
    input1 = torch.FloatTensor(3,4,7)
    input2 = torch.FloatTensor(3,4,5)
    
    asc_prediction_layer = Shared_Sentiment_Prediction(7, 5, 2)
    res = asc_prediction_layer(input1, input2)
        