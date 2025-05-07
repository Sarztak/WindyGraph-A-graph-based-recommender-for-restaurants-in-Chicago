import torch 
import numpy as np 


def generate_recommendation(model, data, user_id, user_mapping, idx_to_restaurant, top_k=10, device='cpu'):
    model.eval()
    data = data.to(device)

    user_idx = user_mapping[user_id]

    with torch.no_grad():
        output = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        user_emb = output['user'][user_id].unsqueeze(0)
        rest_emb = output['restaurant']

        user_rest_edge_type = ('user', 'reviews', 'restaurant')
        edge_index = data[user_rest_edge_type].edge_index

        mask = edge_index[0] == user_idx
        existing_rest_idx = edge_index[1, mask].tolist()

        scores = []
        for r_idx in range(rest_emb.size(0)):
            if r_idx in existing_rest_idx:
                scores.append(-float('inf'))
            else:
                score = model.predict_link(user_emb, rest_emb[r_idx].unsqueeze(0))
                scores.append(score.item())
        
        # get top-k recommendation
        top_indicies = np.argsort(scores)[-top_k:][::-1]

        recommendations = [
            {'restaurant_id': idx_to_restaurant[idx],
             'score': scores[idx]}
             for idx in top_indicies
        ]


        return recommendations
    
    