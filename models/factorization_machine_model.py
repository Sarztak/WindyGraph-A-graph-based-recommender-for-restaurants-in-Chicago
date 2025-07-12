from lightfm import LightFM
from lightfm.data import Dataset
from scipy.sparse import csr_matrix
import pandas as pd

# 1. Load your processed features
reviews_df = pd.read_pickle("data/processed_review_data.pkl")  # one row per (user, restaurant) interaction

# 2. Aggregate User Features
user_df = reviews_df.groupby('user_id').agg({
    'weighted_score': 'mean',
    'weighted_sentiment_score': 'mean',
    'recency_score': 'mean'
}).rename(columns={
    'weighted_score': 'avg_weighted_rating_given',
    'weighted_sentiment_score': 'avg_weighted_sentiment_given',
    'recency_score': 'avg_recency_score'
})

# 3. Aggregate Restaurant Features
item_df = reviews_df.groupby('restaurant_id').agg({
    'weighted_score': 'mean',
    'weighted_sentiment_score': 'mean'
}).rename(columns={
    'weighted_score': 'avg_weighted_rating_received',
    'weighted_sentiment_score': 'avg_weighted_sentiment_received'
})

# Optionally add normalized log review count if available
# item_df['normalized_log_review_count'] = ...

# 4. Build Dataset
dataset = Dataset()
dataset.fit(
    user_df.index,               # all user_ids
    item_df.index                # all restaurant_ids
)

dataset.fit_partial(
    users=user_df.index,
    items=item_df.index,
    user_features=[f"user_{f}" for f in user_df.columns],
    item_features=[f"item_{f}" for f in item_df.columns]
)

# 5. Build Interaction Matrix
interactions = reviews_df.groupby(['user_id', 'restaurant_id']).size().reset_index(name='interaction')
interactions['interaction'] = 1  # binarize
(interaction_matrix, _) = dataset.build_interactions(interactions.itertuples(index=False))

# 6. Build Feature Matrices
user_feature_tuples = [
    (uid, f"user_avg_weighted_rating_given") for uid in user_df.index
] + [
    (uid, f"user_avg_weighted_sentiment_given") for uid in user_df.index
] + [
    (uid, f"user_avg_recency_score") for uid in user_df.index
]

item_feature_tuples = [
    (iid, f"item_avg_weighted_rating_received") for iid in item_df.index
] + [
    (iid, f"item_avg_weighted_sentiment_received") for iid in item_df.index
]

user_features = dataset.build_user_features(user_feature_tuples)
item_features = dataset.build_item_features(item_feature_tuples)

# 7. Train the model
model = LightFM(loss='bpr')  # or 'warp', 'logistic'
model.fit(interaction_matrix, user_features=user_features, item_features=item_features, epochs=10, num_threads=4)


