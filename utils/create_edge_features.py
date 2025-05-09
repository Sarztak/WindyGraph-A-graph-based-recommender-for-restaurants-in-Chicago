import pandas as pd


def build_edge_features(self):
    """Combine all features into edge features for GNN"""
    # Select which features to include in edge representation
    edge_features = self.reviews_df[['restaurant_id', 'user_id', 'rating',
                                    'sentiment_compound', 'sentiment_pos', 'sentiment_neg',
                                    'food_sentiment', 'service_sentiment', 
                                    'price_sentiment', 'ambiance_sentiment',
                                    'user_avg_rating', 'rating_deviation',
                                    'user_review_count', 'user_rating_std',
                                    'avg_word_length', 'sentence_count', 
                                    'avg_sentence_length', 'lexical_diversity']]
    
    # Add topic features if they exist
    topic_cols = [col for col in self.reviews_df.columns if col.startswith('topic_')]
    if topic_cols:
        edge_features = pd.concat([edge_features, self.reviews_df[topic_cols]], axis=1)
    
    return edge_features