import torch
import numpy as np
import logging
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_graph(reviews_df, restaurants_df, embedding_dim=16, seed=42):
    """
    Construct heterogeneous graph from processed data.
    
    Parameters:
    -----------
    reviews_df : DataFrame
        DataFrame containing review data with user, restaurant, ratings, and sentiment info
    restaurants_df : DataFrame
        DataFrame containing restaurant features
    embedding_dim : int
        Dimension for random embeddings for users and categories
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    data : HeteroData
        Heterogeneous graph with user, restaurant, and category nodes
    user_to_id : dict
        Mapping from user IDs to indices
    restaurant_to_id : dict
        Mapping from restaurant IDs to indices
    category_to_id : dict
        Mapping from category IDs to indices
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info("Creating heterogeneous graph from reviews and restaurants data...")
    
    # Initialize HeteroData object
    data = HeteroData()
    
    # ----- Create ID mappings -----
    
    # Create restaurant mapping
    unique_restaurant_ids = restaurants_df['id'].unique()
    restaurant_to_id = {rest_id: idx for idx, rest_id in enumerate(unique_restaurant_ids)}
    logger.info(f"Created mapping for {len(restaurant_to_id)} restaurants")
    
    # Create user mapping
    unique_user_ids = reviews_df['user_id'].unique()
    user_to_id = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
    logger.info(f"Created mapping for {len(user_to_id)} users")
    
    # Create category mapping
    all_categories = set()
    for categories in restaurants_df['categories_list']:
        if isinstance(categories, list):
            all_categories.update(categories)
        else:
            logger.warning(f"Found non-list category format: {type(categories)}. Skipping.")
    
    unique_categories = sorted(list(all_categories))
    category_to_id = {category: idx for idx, category in enumerate(unique_categories)}
    logger.info(f"Created mapping for {len(category_to_id)} categories")
    
    # ----- Create Node Features -----
    
    # User node features - random initialization
    num_users = len(user_to_id)
    data['user'].x = torch.randn((num_users, embedding_dim), dtype=torch.float)
    
    # Restaurant node features with proper sorting by index
    try:
        restaurant_features_sorted = [None] * len(restaurant_to_id)
        for rest_id, idx in restaurant_to_id.items():
            restaurant_row = restaurants_df[restaurants_df['id'] == rest_id]
            
            if restaurant_row.empty:
                raise ValueError(f"Restaurant ID {rest_id} not found in restaurants DataFrame")
                
            # Required columns
            required_columns = [
                'normalized_rating', 'normalized_wilson_score', 
                'normalized_latitude', 'normalized_longitude', 
                'popularity_score', 'normalized_log_review_count'
            ]
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in restaurant_row.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in restaurants DataFrame: {missing_columns}")
            
            restaurant_features_sorted[idx] = [
                float(restaurant_row['normalized_rating'].iloc[0]),
                float(restaurant_row['normalized_wilson_score'].iloc[0]),
                float(restaurant_row['normalized_latitude'].iloc[0]),
                float(restaurant_row['normalized_longitude'].iloc[0]),
                float(restaurant_row['popularity_score'].iloc[0]),
                float(restaurant_row['log_review_count'].iloc[0])
            ]
        
        data['restaurant'].x = torch.tensor(restaurant_features_sorted, dtype=torch.float)
        logger.info(f"Created features for {len(restaurant_features_sorted)} restaurants")
    except Exception as e:
        logger.error(f"Error creating restaurant features: {str(e)}")
        raise
    
    # Category node features - random initialization
    num_categories = len(category_to_id)
    data['category'].x = torch.randn((num_categories, embedding_dim), dtype=torch.float)
    
    # ----- Create Edges -----
    
    # User-Restaurant edges (from reviews)
    user_to_rest_edges = []
    user_rest_edge_features = []
    
    # Required review columns
    required_review_columns = [
        'user_id', 'restaurant_id', 'normalized_rating', 
        'bert_score', 'recency_score', 'weighted_rating_score', 
        'weighted_sentiment_score'
    ]
    
    # Check if all required columns exist
    missing_columns = [col for col in required_review_columns if col not in reviews_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in reviews DataFrame: {missing_columns}")
    
    try:
        skipped_reviews = 0
        for _, review in reviews_df.iterrows():
            if review['user_id'] in user_to_id and review['restaurant_id'] in restaurant_to_id:
                u_idx = user_to_id[review['user_id']]
                r_idx = restaurant_to_id[review['restaurant_id']]
                
                # Edge index [source, target]
                user_to_rest_edges.append([u_idx, r_idx])
                
                # Edge features
                edge_feat = [
                    float(review['normalized_rating']),
                    float(review['bert_score']),
                    float(review['recency_score']),
                    float(review['weighted_rating_score']),
                    float(review['weighted_sentiment_score'])
                ]
                user_rest_edge_features.append(edge_feat)
            else:
                skipped_reviews += 1
        
        if skipped_reviews > 0:
            logger.warning(f"Skipped {skipped_reviews} reviews due to missing user or restaurant IDs")
        
        # Convert to PyTorch tensors if edges exist
        if not user_to_rest_edges:
            raise ValueError("No valid user-restaurant edges found")
            
        user_to_rest_edges = torch.tensor(user_to_rest_edges).t().contiguous()  # Shape [2, num_edges]
        user_rest_edge_features = torch.tensor(user_rest_edge_features, dtype=torch.float)
        
        # Add edges to the graph
        data['user', 'reviews', 'restaurant'].edge_index = user_to_rest_edges
        data['user', 'reviews', 'restaurant'].edge_attr = user_rest_edge_features
        
        # Create reverse edges
        rest_to_user_edges = user_to_rest_edges.flip(0)  # Swap source & target
        
        data['restaurant', 'reviewed_by', 'user'].edge_index = rest_to_user_edges
        data['restaurant', 'reviewed_by', 'user'].edge_attr = user_rest_edge_features  # Reuse features
        
        logger.info(f"Created {user_to_rest_edges.shape[1]} user-restaurant edges")
    except Exception as e:
        logger.error(f"Error creating user-restaurant edges: {str(e)}")
        raise
    
    # Restaurant-Category edges
    try:
        rest_to_cat_edges = []
        skipped_categories = 0
        
        for _, restaurant in restaurants_df.iterrows():
            if restaurant['id'] in restaurant_to_id:
                r_idx = restaurant_to_id[restaurant['id']]
                
                # Process categories list (already a list, not a string representation)
                categories = restaurant['categories_list']
                if isinstance(categories, list):
                    for category in categories:
                        if category in category_to_id:
                            c_idx = category_to_id[category]
                            rest_to_cat_edges.append([r_idx, c_idx])
                        else:
                            skipped_categories += 1
                else:
                    logger.warning(f"Skipping non-list categories for restaurant {restaurant['id']}")
        
        if skipped_categories > 0:
            logger.warning(f"Skipped {skipped_categories} categories not found in the mapping")
        
        # Convert to PyTorch tensors if edges exist
        if not rest_to_cat_edges:
            logger.warning("No valid restaurant-category edges found")
        else:
            rest_to_cat_edges = torch.tensor(rest_to_cat_edges).t().contiguous()
            
            # Create reverse edges
            cat_to_rest_edges = rest_to_cat_edges.flip(0)
            
            # Add edges to the graph
            data['restaurant', 'belongs_to', 'category'].edge_index = rest_to_cat_edges
            data['category', 'has', 'restaurant'].edge_index = cat_to_rest_edges
            
            logger.info(f"Created {rest_to_cat_edges.shape[1]} restaurant-category edges")
    except Exception as e:
        logger.error(f"Error creating restaurant-category edges: {str(e)}")
        raise
    
    logger.info("Graph creation completed successfully")
    return data, user_to_id, restaurant_to_id, category_to_id

def split_graph_edges(data, val_ratio=0.1, test_ratio=0.2, neg_sampling_ratio=1.0, seed=42):
    """
    Split edges for training, validation and testing using RandomLinkSplit.
    
    Parameters:
    -----------
    data : HeteroData
        The graph to split
    val_ratio : float
        Ratio of edges to use for validation
    test_ratio : float
        Ratio of edges to use for testing
    neg_sampling_ratio : float
        Ratio of negative samples to positive samples
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    train_data : HeteroData
        Graph with training edges
    val_data : HeteroData
        Graph with validation edges
    test_data : HeteroData
        Graph with test edges
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    logger.info("Splitting graph edges for training, validation, and testing...")
    
    try:
        # Define edge types for splitting
        edge_types = [('user', 'reviews', 'restaurant')]
        rev_edge_types = [('restaurant', 'reviewed_by', 'user')]
        
        # Create the transform
        transform = RandomLinkSplit(
            num_val=val_ratio,
            num_test=test_ratio,
            neg_sampling_ratio=neg_sampling_ratio,
            edge_types=edge_types,
            rev_edge_types=rev_edge_types,
            is_undirected=False  # This is a directed graph
        )
        
        # Apply the transform
        train_data, val_data, test_data = transform(data)
        
        # Log split sizes
        logger.info(f"Training edges: {train_data[edge_types[0]].edge_index.shape[1]}")
        logger.info(f"Validation edges: {val_data[edge_types[0]].edge_index.shape[1]}")
        logger.info(f"Testing edges: {test_data[edge_types[0]].edge_index.shape[1]}")
        
        return train_data, val_data, test_data
    except Exception as e:
        logger.error(f"Error during graph splitting: {str(e)}")
        raise

# Example usage:
"""
import pandas as pd

# Load your reviews and restaurants DataFrames
reviews_df = pd.read_csv('processed_reviews.csv')
restaurants_df = pd.read_csv('processed_restaurants.csv')

# Create graph
graph, user_to_id, restaurant_to_id, category_to_id = create_graph(
    reviews_df, restaurants_df, embedding_dim=16
)

# Split edges for training
train_data, val_data, test_data = split_graph_edges(
    graph, val_ratio=0.1, test_ratio=0.2, neg_sampling_ratio=1.0
)

# Save the graph and mappings
import pickle
with open('graph_data.pkl', 'wb') as f:
    pickle.dump({
        'graph': graph,
        'user_to_id': user_to_id,
        'restaurant_to_id': restaurant_to_id,
        'category_to_id': category_to_id,
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }, f)
"""