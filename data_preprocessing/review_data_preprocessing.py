import json
import torch
import pandas as pd
from datasets import Dataset
from datetime import datetime
from transformers import pipeline

def flatten_restaurant_reviews(json_data):
    """
    Flatten restaurant reviews from a nested JSON structure into a pandas DataFrame.
    Each row will contain a review with its associated restaurant ID.
    """
    flattened_data = []
    
    # Iterate through each restaurant and its reviews
    for restaurant_id, reviews in json_data.items():
        for review in reviews:
            # Create a flat record with restaurant_id and all review data
            flat_record = {
                'restaurant_id': restaurant_id,
                'review_id': review['id'],
                'rating': review['rating'],
                'text': review['text'],
                'time_created': review['time_created'],
                'user_id': review['user']['id'],
                'user_name': review['user']['name'],
            }
            flattened_data.append(flat_record)
    
    # Convert to DataFrame
    return pd.DataFrame(flattened_data)

def add_user_stats(reviews_df):
    """Add user-level statistics as features"""
    # Calculate average rating per user
    user_avg_rating = reviews_df.groupby('user_id')['rating'].mean()
    reviews_df['user_avg_rating'] = reviews_df['user_id'].map(user_avg_rating)
    
    # Calculate rating deviation (how this rating differs from user's average)
    reviews_df['rating_deviation'] = reviews_df['rating'] - reviews_df['user_avg_rating']
    
    # Count reviews per user
    user_review_count = reviews_df.groupby('user_id')['review_id'].count()
    reviews_df['user_review_count'] = reviews_df['user_id'].map(user_review_count)
    
    # Calculate standard deviation of user ratings (consistent vs variable rater)
    user_rating_std = reviews_df.groupby('user_id')['rating'].std()
    reviews_df['user_rating_std'] = reviews_df['user_id'].map(user_rating_std)
    
    return reviews_df

def add_recency_score(reviews_df):
    reviews_df['time_created'] = pd.to_datetime(reviews_df['time_created'])
    current_time = datetime.now()
    reviews_df['days_since_review'] = (current_time - reviews_df['time_created']).dt.total_seconds() / (60*60*24)
    max_days = reviews_df['days_since_review'].max()
    reviews_df['recency_score'] = 1 - (reviews_df['days_since_review'] / max_days)

    return reviews_df

def add_normalized_rating(reviews_df):
    min_rating = reviews_df['rating'].min()
    max_rating = reviews_df['rating'].max()

    # Min-max normalization
    reviews_df['normalized_rating'] = (reviews_df['rating'] - min_rating) / max_rating 
    
    return reviews_df

def add_weighted_score(reviews_df, alpha=0.5):
    reviews_df['weighted_score'] = alpha * reviews_df['normalized_rating'] + (1 - alpha) * reviews_df['recency_score']

    return reviews_df

from transformers import pipeline

def add_bert_sentiment(reviews_df):
    """
    Add sentiment scores using a pre-trained BERT model.
    Uses Hugging Face pipeline for sentiment analysis.
    """

    device = 0 if torch.cuda.is_available() else -1

    # Convert the DataFrame to a Hugging Face Dataset
    dataset = Dataset.from_pandas(reviews_df)

    # Initialize the sentiment analysis pipeline with GPU support
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device  # Use GPU if available, otherwise fallback to CPU
    )

    # Define a function to apply the sentiment analysis in batches
    def analyze_sentiment(batch):
        sentiments = sentiment_analyzer(batch['text'])
        batch['bert_sentiment'] = [sent['label'] for sent in sentiments]
        batch['bert_score'] = [sent['score'] for sent in sentiments]
        return batch

    # Apply the function to the dataset in batches
    dataset = dataset.map(analyze_sentiment, batched=True, batch_size=16)

    # Convert the dataset back to a DataFrame
    reviews_df = dataset.to_pandas()

    # Map BERT output to positive, neutral, negative
    def map_sentiment(label):
        if label in ['4 stars', '5 stars']:
            return 'positive'
        elif label == '3 stars':
            return 'neutral'
        else:
            return 'negative'

    # Calculate a weighted sentiment score (similar to compound)
    def calculate_weighted_score(label, score):
        stars = int(label[0])  # Extract the numeric part from '4 stars', '5 stars', etc.
        return (stars / 5) * score

    # Add the mapped sentiment and weighted score to the DataFrame
    reviews_df['mapped_sentiment'] = reviews_df['bert_sentiment'].apply(map_sentiment)
    reviews_df['weighted_score'] = reviews_df.apply(lambda x: calculate_weighted_score(x['bert_sentiment'], x['bert_score']), axis=1)

    return reviews_df



def preprocess_reviews(reviews_df):

    # initialize extract textual features from the reviews posted by users
    # not useful
    # text_feature_extractor = TextFeatureExtractor(reviews_df=reviews_df)

    # did not add aspect sentiments, topic modelling and user stats because they have harder
    # assumptions to make also because that is added complexity and may not yield results for this 
    # current first iteration of project

    reviews_df = add_bert_sentiment(reviews_df)
    reviews_df = add_recency_score(reviews_df)
    reviews_df = add_normalized_rating(reviews_df)
    reviews_df = add_weighted_score(reviews_df)
    
    return reviews_df


if __name__ == "__main__":
    with open(r'data/reviews.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. Convert to DataFrame
    reviews_df = flatten_restaurant_reviews(data)
    reviews_df = preprocess_reviews(reviews_df)


    # save the file
    reviews_df.to_pickle('data/processed_review_data.pkl')
    print(reviews_df.head())
    print(reviews_df.columns)
