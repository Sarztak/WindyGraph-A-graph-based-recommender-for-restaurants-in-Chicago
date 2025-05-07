import torch 
import torch.nn.functional as F 
from torch_geometric.utils import negative_sampling

def train_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    
    user_rest_edge_type = ('user', 'reviews', 'restaurant')

    pos_edge_index = data[user_rest_edge_type].edge_index 

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(data['user'].x.size(0), data['restaurant'].x.size(0)),
        num_neg_samples=pos_edge_index.size(1)
    )


    output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

    user_emb = output['user']
    rest_emb = output['restaurant']

    pos_u, pos_r = pos_edge_index
    pos_pred = model.predict_link(user_emb[pos_u], rest_emb[pos_r])
    pos_loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred))


    neg_u, neg_r = neg_edge_index
    neg_pred = model.predict_link(user_emb[neg_u], rest_emb[neg_r])
    neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))

    loss = pos_loss + neg_loss

    loss.bachward()
    optimizer.step()

    return loss.item()
