import torch
from torch_geometric import HeteroData

def create_graph(
        user_features, restaurant_features, category_features,
        reviews_df, restaurant_categories_df, user_mapping, restaurant_mapping,
        category_mapping, topic_distribution
):
    '''Construct Heterogenous Graph from Processed Data
        processed_restaurant_columns = [
        'id', 'name', 'rating', 'review_count', 'latitude', 'longitude',
       'categories_list', 'log_review_count', 'normalized_rating',
       'popularity_score', 'wilson_score', 'normalized_wilson_score',
       'normalized_latitude', 'normalized_longitude']

       processed_reviews_columns = [
       'restaurant_id', 'review_id', 'rating', 'text', 'time_created',
       'user_id', 'user_name']
    '''

    # Initialize HeteroData object
    data = HeteroData()

    # add user, restaurant and category node features
    data['user'].x = user_features
    data['restaurant'].x = restaurant_features
    data['category'].x = category_features


    # Process user-restaurant edges
    user_to_rest_edges = []
    user_to_rest_edge_features = []

    for idx, review in reviews_df.iterrows():
        
        # pull out index from the user_id in each row   
        u_idx = user_mapping[review['user_id']]

        # pull out index from the restaurant_id in each row
        r_idx = restaurant_mapping[review['restaurant_id']]

        user_to_rest_edges.append([u_idx, r_idx])

        edge_features = [
            
        ]
