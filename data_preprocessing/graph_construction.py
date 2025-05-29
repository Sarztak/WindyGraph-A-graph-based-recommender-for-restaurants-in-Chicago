import logging
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import RandomLinkSplit

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Required dataframe columns
# -----------------------------------------------------------------------------
REQUIRED_REVIEW_COLS = {
    'user_id', 'restaurant_id',
    'normalized_rating', 'bert_score', 'recency_score',
    'weighted_rating_score', 'weighted_sentiment_score',
}

REQUIRED_RESTAURANT_COLS = {
    'id', 'categories_list',
    'normalized_rating_restaurants', 'normalized_wilson_score',
    'normalized_latitude', 'normalized_longitude',
    'popularity_score', 'normalized_log_review_count',
}

# -----------------------------------------------------------------------------
# Helper validation
# -----------------------------------------------------------------------------

def _validate_inputs(reviews_df, restaurants_df) -> None:
    """Ensure dataframes contain required columns and are non‑empty."""
    missing_r_cols = REQUIRED_REVIEW_COLS.difference(reviews_df.columns)
    missing_rest_cols = REQUIRED_RESTAURANT_COLS.difference(restaurants_df.columns)

    if missing_r_cols:
        raise ValueError(f"`reviews_df` missing columns: {missing_r_cols}")
    if missing_rest_cols:
        raise ValueError(f"`restaurants_df` missing columns: {missing_rest_cols}")
    if reviews_df.empty:
        raise ValueError("`reviews_df` is empty – cannot build graph.")
    if restaurants_df.empty:
        raise ValueError("`restaurants_df` is empty – cannot build graph.")


# -----------------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------------

def create_graph(
    reviews_df,
    restaurants_df,
    embedding_dim: int = 16,
    seed: int = 84,
) -> Tuple[HeteroData, Dict[Any, int], Dict[Any, int], Dict[str, int]]:
    """Build a PyG HeteroData graph with users, restaurants and categories."""

    _validate_inputs(reviews_df, restaurants_df)

    torch.manual_seed(seed)
    np.random.seed(seed)

    data = HeteroData()

    # ------------------------------------------------------------------
    # ID mappings
    # ------------------------------------------------------------------
    unique_restaurant_ids = restaurants_df['id'].unique()
    restaurant_to_id: Dict[Any, int] = {rid: idx for idx, rid in enumerate(unique_restaurant_ids)}

    unique_user_ids = reviews_df['user_id'].unique()
    user_to_id: Dict[Any, int] = {uid: idx for idx, uid in enumerate(unique_user_ids)}

    # Categories
    categories_set: set[str] = set()
    for cats in restaurants_df['categories_list']:
        if isinstance(cats, (list, tuple, set)):
            categories_set.update(cats)
        else:
            logger.warning("Unexpected categories_list type: %s", type(cats))
    if not categories_set:
        raise ValueError("No categories extracted from restaurants dataframe.")
    unique_categories: List[str] = sorted(categories_set)
    category_to_id: Dict[str, int] = {c: idx for idx, c in enumerate(unique_categories)}

    # ------------------------------------------------------------------
    # Node features
    # ------------------------------------------------------------------
    data['user'].x = torch.randn((len(user_to_id), embedding_dim), dtype=torch.float)
    data['category'].x = torch.randn((len(category_to_id), embedding_dim), dtype=torch.float)

    restaurant_features: List[List[float] | None] = [None] * len(restaurant_to_id)
    for rest_id, idx in restaurant_to_id.items():
        row = restaurants_df.loc[restaurants_df['id'] == rest_id]
        assert not row.empty, (
            f"Restaurant id {rest_id} present in mapping but missing in dataframe. "
            "Ensure preprocessing is consistent."
        )
        restaurant_features[idx] = [
            float(row['normalized_rating_restaurants'].iloc[0]),
            float(row['normalized_wilson_score'].iloc[0]),
            float(row['normalized_latitude'].iloc[0]),
            float(row['normalized_longitude'].iloc[0]),
            float(row['popularity_score'].iloc[0]),
            float(row['normalized_log_review_count'].iloc[0]),
        ]
    assert all(f is not None for f in restaurant_features), "Unpopulated restaurant feature entry detected."
    data['restaurant'].x = torch.tensor(restaurant_features, dtype=torch.float)

    # ------------------------------------------------------------------
    # Edges: User -> Restaurant (reviews)
    # ------------------------------------------------------------------
    user_rest_edges, user_rest_attr = [], []
    for _, rev in reviews_df.iterrows():
        try:
            u_idx = user_to_id[rev['user_id']]
            r_idx = restaurant_to_id[rev['restaurant_id']]
        except KeyError as e:
            logger.warning("Skipping review with unmapped ID: %s", e)
            continue
        user_rest_edges.append([u_idx, r_idx])
        user_rest_attr.append([
            float(rev['normalized_rating']),
            float(rev['bert_score']),
            float(rev['recency_score']),
            float(rev['weighted_rating_score']),
            float(rev['weighted_sentiment_score']),
        ])
    assert user_rest_edges, "No user-restaurant edges created."

    user_rest_edge_index = torch.tensor(user_rest_edges, dtype=torch.long).t().contiguous()
    user_rest_edge_attr = torch.tensor(user_rest_attr, dtype=torch.float)

    data[('user', 'reviews', 'restaurant')].edge_index = user_rest_edge_index
    data[('user', 'reviews', 'restaurant')].edge_attr = user_rest_edge_attr
    data[('restaurant', 'reviewed_by', 'user')].edge_index = user_rest_edge_index.flip(0)
    data[('restaurant', 'reviewed_by', 'user')].edge_attr = user_rest_edge_attr

    # ------------------------------------------------------------------
    # Edges: Restaurant -> Category (belongs_to)
    # ------------------------------------------------------------------
    rest_cat_edges = []
    for _, row in restaurants_df.iterrows():
        r_idx = restaurant_to_id[row['id']]
        cats = row['categories_list'] if isinstance(row['categories_list'], (list, tuple, set)) else []
        for c in cats:
            if c not in category_to_id:
                logger.warning("Category '%s' not in mapping - skipping.", c)
                continue
            rest_cat_edges.append([r_idx, category_to_id[c]])

    assert rest_cat_edges, "No restaurant - category edges created - check categories_list preprocessing."

    rest_cat_edge_index = torch.tensor(rest_cat_edges, dtype=torch.long).t().contiguous()
    data[('restaurant', 'belongs_to', 'category')].edge_index = rest_cat_edge_index
    data[('category', 'has', 'restaurant')].edge_index = rest_cat_edge_index.flip(0)

    logger.info(
        "Graph assembled - Users: %d | Restaurants: %d | Categories: %d | Review edges: %d | RC edges: %d",
        len(user_to_id), len(restaurant_to_id), len(category_to_id),
        user_rest_edge_index.size(1), rest_cat_edge_index.size(1)
    )

    return data, user_to_id, restaurant_to_id, category_to_id


# -----------------------------------------------------------------------------
# Split + loaders
# -----------------------------------------------------------------------------

def build_loaders(
    graph: HeteroData,
    batch_size: int = 64,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    disjoint_train_ratio: float = 0.3,
    neg_sampling_ratio: float = 1.0,
    seed: int = 42,
):
    """Return train/val/test LinkNeighborLoaders after RandomLinkSplit on review edges."""

    torch.manual_seed(seed)

    splitter = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        disjoint_train_ratio=disjoint_train_ratio,
        neg_sampling_ratio=neg_sampling_ratio,
        add_negative_train_samples=False,
        edge_types=[('user', 'reviews', 'restaurant')],  # we only split review edges
        rev_edge_types=[('restaurant', 'reviewed_by', 'user')],
    )

    train_data, val_data, test_data = splitter(graph)
    logger.info(
        "Edge split - train: %d | val: %d | test: %d",
        train_data[('user', 'reviews', 'restaurant')].edge_index.size(1),
        val_data[('user', 'reviews', 'restaurant')].edge_index.size(1),
        test_data[('user', 'reviews', 'restaurant')].edge_index.size(1),
    )

    loader_kwargs = dict(num_neighbors=[20, 10], batch_size=batch_size, shuffle=False)

    train_loader = LinkNeighborLoader(
        data=train_data,
        edge_label_index=train_data[('user', 'reviews', 'restaurant')].edge_index,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = LinkNeighborLoader(
        data=val_data,
        edge_label_index=val_data[('user', 'reviews', 'restaurant')].edge_index,
        **loader_kwargs,
    )
    test_loader = LinkNeighborLoader(
        data=test_data,
        edge_label_index=test_data[('user', 'reviews', 'restaurant')].edge_index,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# -----------------------------------------------------------------------------
# Example usage (commented)
# -----------------------------------------------------------------------------
import pandas as pd
reviews_df = pd.read_csv('processed_reviews.csv')
restaurants_df = pd.read_csv('processed_restaurants.csv')
graph, u_map, r_map, c_map = create_graph(reviews_df, restaurants_df)
train_loader, val_loader, test_loader = build_loaders(graph)
