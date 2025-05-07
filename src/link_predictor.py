import torch
import torch.nn.functional as F 
class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, user_emb, restaurant_emb):

        x = torch.cat([user_emb, restaurant_emb], dim=-1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        return torch.sigmoid(x)