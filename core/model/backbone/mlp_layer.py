import torch
 
class MLP_layer(torch.nn.Module):
    
 
    def __init__(self,in_dim=10):
        super(MLP_layer,self).__init__()
        self.linear1=torch.nn.Linear(in_dim,in_dim)
        self.bn = torch.nn.BatchNorm1d(in_dim)
        self.relu=torch.nn.ReLU()
  
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
