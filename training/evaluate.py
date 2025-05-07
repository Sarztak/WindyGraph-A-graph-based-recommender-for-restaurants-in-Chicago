import torch 
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling 

def precision_at_k(y_true, y_pred, k):
    top_k = torch.topk(y_pred, k).indices
    hits = y_true[top_k].sum().item()
    return hits / k

def recall_at_k(y_true, y_pred, k):
    top_k = torch.topk(y_pred, k).indices
    hits = y_true[top_k].sum().item()
    return hits / y_true.sum().item()

def ndcg_at_k(y_true, y_pred, k):
    top_k = torch.topk(y_pred, k).indices
    dcg = (y_true[top_k] / torch.log2(torch.arange(2, k + 2).float())).sum().item()
    idcg = (torch.sort(y_true, descending=True).values[:k] / torch.log2(torch.arange(2, k + 2).float())).sum().item()
    return dcg / idcg if idcg > 0 else 0

def mean_reciprocal_rank(y_true, y_pred):
    sorted_indices = torch.argsort(y_pred, descending=True)
    ranks = (torch.nonzero(y_true[sorted_indices]) + 1).float()
    if len(ranks) == 0:
        return 0.0
    return (1.0 / ranks[0]).item()

def hit_rate_at_k(y_true, y_pred, k):
    top_k = torch.topk(y_pred, k).indices
    hits = y_true[top_k].sum().item()
    return 1.0 if hits > 0 else 0.0

def evaluate_model(model, data, pos_edge_index, device, k=10):
    model.eval()
    data = data.to(device)

    user_rest_edge_type = ('user', 'reviews', 'restaurant')

    pos_edge_index = data[user_rest_edge_type].edge_index 

    neg_edge_index = negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=(data['user'].x.size(0), data['restaurant'].x.size(0)),
        num_neg_samples=pos_edge_index.size(1)
    )

    with torch.no_grad():
        output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)

        user_emb = output['user']
        rest_emb = output['restaurant']

        pos_u, pos_r = pos_edge_index
        pos_pred = model.predict_link(user_emb[pos_u], rest_emb[pos_r])

        neg_u, neg_r = neg_edge_index
        neg_pred = model.predict_link(user_emb[neg_u], rest_emb[neg_r])

        y_true = torch.cat([torch.ones_like(pos_pred), torch.zeros_like(neg_pred)])
        y_pred = torch.cat([pos_pred, neg_pred])

        # Calculate metrics
        auc = roc_auc_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        ap = average_precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
        precision = precision_at_k(y_true, y_pred, k)
        recall = recall_at_k(y_true, y_pred, k)
        ndcg = ndcg_at_k(y_true, y_pred, k)
        mrr = mean_reciprocal_rank(y_true, y_pred)
        hr = hit_rate_at_k(y_true, y_pred, k)

        # Leave-One-Out Evaluation (Gold Standard)
        loo_hr = []
        loo_mrr = []

        for user_id in torch.unique(pos_u):
            user_mask = pos_u == user_id
            user_true = y_true[user_mask]
            user_pred = y_pred[user_mask]

            # Sort the scores
            sorted_indices = torch.argsort(user_pred, descending=True)
            user_true = user_true[sorted_indices]

            # Hit Rate and MRR for Leave-One-Out
            loo_hr.append(hit_rate_at_k(user_true, user_pred, k))
            loo_mrr.append(mean_reciprocal_rank(user_true, user_pred))

        avg_loo_hr = np.mean(loo_hr)
        avg_loo_mrr = np.mean(loo_mrr)

    return {
        'AUC': auc,
        'Average Precision': ap,
        'Precision@{}'.format(k): precision,
        'Recall@{}'.format(k): recall,
        'NDCG@{}'.format(k): ndcg,
        'MRR': mrr,
        'Hit Rate@{}'.format(k): hr,
        'LOO HR@{}'.format(k): avg_loo_hr,
        'LOO MRR': avg_loo_mrr
    }


