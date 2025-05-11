import torch
import torch.nn.functional as F 
from torch_geometric.nn import HeteroConv, GATConv, SAGEConv

class LinkPredictor(torch.nn.Module):
    """
    Predicts links between users and restaurants by scoring 
    the concatenation of their embeddings.
    """
    def __init__(self, hidden_channels):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)
        self.dropout = torch.nn.Dropout(0.2)
        
    def forward(self, user_emb, restaurant_emb):
        # Concatenate user and restaurant embeddings
        x = torch.cat([user_emb, restaurant_emb], dim=-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # Return score (no sigmoid here, we'll apply it in the loss function)
        return x

class RestaurantRecommenderGNN(torch.nn.Module):
    def __init__(self, hidden_channels, edge_feature_dim=5):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_feature_dim = edge_feature_dim
        self.node_types = ['user', 'restaurant', 'category']

        # First convolutional layer
        self.conv1 = HeteroConv({
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
                (-1, -1),  # Fixed dimension
                self.hidden_channels
            ),
            ('category', 'has', 'restaurant'): SAGEConv(
                (-1, -1),  # Fixed dimension
                self.hidden_channels
            )
        })

        # Second convolutional layer
        self.conv2 = HeteroConv({
            ('user', 'reviews', 'restaurant'): GATConv(
                self.hidden_channels, 
                self.hidden_channels, 
                edge_dim=self.edge_feature_dim,
                add_self_loops=True,
                dropout=0.2,
                bias=True
            ),
            ('restaurant', 'reviewed_by', 'user'): GATConv(
                self.hidden_channels, 
                self.hidden_channels, 
                edge_dim=self.edge_feature_dim,
                add_self_loops=True,
                dropout=0.2,
                bias=True
            ),
            ('restaurant', 'belongs_to', 'category'): SAGEConv(
                (self.hidden_channels, self.hidden_channels),
                self.hidden_channels
            ),
            ('category', 'has', 'restaurant'): SAGEConv(
                (self.hidden_channels, self.hidden_channels),
                self.hidden_channels
            )
        })

        # Batch normalization for each node type
        self.batch_norm1 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in self.node_types
        })

        self.batch_norm2 = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels)
            for node_type in self.node_types
        })

        # Link predictor for recommendation
        self.link_predictor = LinkPredictor(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass through the GNN.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            edge_attr_dict: Optional dictionary of edge attributes for each edge type
        
        Returns:
            Dictionary of node embeddings for each node type
        """
        # Process edge attributes - only review edges have attributes
        if edge_attr_dict is None:
            edge_attr_dict = {}
        
        # First layer
        x_dict_1 = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        
        # Apply batch normalization and activation to each node type
        x_dict_1 = {
            key: F.relu(self.batch_norm1[key](x))
            for key, x in x_dict_1.items() if key in self.node_types
        }
        
        # Second layer
        x_dict_2 = self.conv2(x_dict_1, edge_index_dict, edge_attr_dict)
        
        # Apply batch normalization and activation to each node type
        x_dict_2 = {
            key: F.relu(self.batch_norm2[key](x))
            for key, x in x_dict_2.items() if key in self.node_types
        }
        
        return x_dict_2
    
    def predict_link(self, user_emb, restaurant_emb):
        """
        Predict the likelihood of a link between user and restaurant.
        
        Args:
            user_emb: User embedding tensor
            restaurant_emb: Restaurant embedding tensor
            
        Returns:
            Score for the potential link
        """
        return self.link_predictor(user_emb, restaurant_emb)