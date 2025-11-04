from torch import nn 
import torch.nn.functional as F


class WineClassifier(nn.Module):
    def __init__(self, input_shape, hidden_layer_1, hidden_layer_2, num_classes, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_shape, hidden_layer_1)
        self.bn1 = nn.BatchNorm1d(hidden_layer_1)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear2 = nn.Linear(hidden_layer_1, hidden_layer_2)
        self.bn2 = nn.BatchNorm1d(hidden_layer_2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear3 = nn.Linear(hidden_layer_2, num_classes)

    def forward(self, x):
        # Flatten if needed (aunque para vinos ya viene plano)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.bn1(self.linear1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.linear2(x)))
        x = self.dropout2(x)
        
        x = self.linear3(x)  # logits para clasificación
        return x
