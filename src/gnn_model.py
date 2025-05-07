import torch
import torch.nn.functional as F 
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv
from .link_predictor import LinkPredictor

'''
Do I need 
1. Batchnorm: yes, improve convergence
2. Dropout: yes, prevent model from memorizing most frequent recommendations
3. Skip connections: No, only two layers; complicated to add; need to check dimes
4. bias term: yes, can be helpful
5. self loops: yes, if I want nodes to retain some of their original features.
'''

class RestaurantRecommenderGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_feature_dim):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_feature_dim = edge_feature_dim
        self.node_types = ['user', 'restaurant', 'category']

        # first convolutional layer
        self.conv1 = HeteroConv(
            {
                ('user', 'reviews', 'restaurant'): GATConv(
                    (-1, -1), 
                    self.hidden_channels, 
                    edge_dim=self.edge_feature_dim,
                    add_self_loops=True,
                    dropout=0.2,
                    bias=True
                ),

                ('restaurant', 'reviewed_by', 'user'): GATConv(
                    (-1, -1), 
                    self.hidden_channels, 
                    edge_dim=self.edge_feature_dim,
                    add_self_loops=True,
                    dropout=0.2,
                    bias=True
                ),
                
                ('restaurant', 'belongs_to', 'category'): SAGEConv(
                    (-1, 1),
                    self.hidden_channels
                ),

                ('category', 'has', 'restaurant'): SAGEConv(
                    (-1, 1),
                    self.hidden_channels
                )
            }
        )

        #  second convolutional layer
        self.conv2 = HeteroConv(
            {
                ('user', 'reviews', 'restaurant'): GATConv(
                    (-1, -1), 
                    self.hidden_channels, 
                    edge_dim=self.edge_feature_dim,
                    add_self_loops=True,
                    dropout=0.2,
                    bias=True
                ),

                ('restaurant', 'reviewed_by', 'user'): GATConv(
                    (-1, -1), 
                    self.hidden_channels, 
                    edge_dim=self.edge_feature_dim,
                    add_self_loops=True,
                    dropout=0.2,
                    bias=True
                ),
                
                ('restaurant', 'belongs_to', 'category'): SAGEConv(
                    (-1, 1),
                    self.hidden_channels
                ),

                ('category', 'has', 'restaurant'): SAGEConv(
                    (-1, 1),
                    self.hidden_channels
                ),
            }
        )

        self.batch_norm1 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in self.node_types
        })

        self.batch_norm2 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in self.node_types
        })


        self.link_predictor = LinkPredictor(hidden_channels)


    def forward(self, x_dict, edge_index_dict, edge_attr_dict):

        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)

        x_dict = {
            key: F.relu(self.batch_norm1[key](x))
            for key, x in self.x_dict if key in self.node_types
        }

        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)

        x_dict = {
            key: F.relu(self.batch_norm2[key](x))
            for key, x in self.x_dict if key in self.node_types
        }

        return x_dict
    
    def predict_link(self, user_emb, restaurant_emb):
        return self.link_predictor(user_emb, restaurant_emb)
        
