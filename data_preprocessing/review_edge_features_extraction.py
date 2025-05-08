import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Import our text preprocessor
from text_preprocessing import TextPreprocessor

class ReviewFeatureExtractor:
    def __init__(self, reviews_df):
        """Initialize with the flattened dataframe of reviews"""
        self.reviews_df = reviews_df
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()
        # Preprocess the text
        self.reviews_df = self.preprocessor.process_reviews_df(self.reviews_df)
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize BERT for review embeddings
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def add_sentiment_scores(self):
        """Add sentiment analysis scores using cleaned text"""
        # Apply sentiment analysis to each review using the text cleaned for sentiment
        sentiment_scores = self.reviews_df['sentiment_text'].apply(
            lambda x: self.sia.polarity_scores(x)
        )
        
        # Extract individual sentiment components
        self.reviews_df['sentiment_neg'] = sentiment_scores.apply(lambda x: x['neg'])
        self.reviews_df['sentiment_neu'] = sentiment_scores.apply(lambda x: x['neu'])
        self.reviews_df['sentiment_pos'] = sentiment_scores.apply(lambda x: x['pos'])
        self.reviews_df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['compound'])
        
        return self
    
    def extract_aspect_sentiments(self):
        """Extract sentiment for specific aspects of restaurant reviews"""
        # Define aspects to look for
        aspects = {
            'food': ['food', 'taste', 'flavor', 'delicious', 'dish', 'menu', 'portion', 'appetizer', 'dessert'],
            'service': ['service', 'staff', 'waiter', 'waitress', 'server', 'employee', 'manager', 'host', 'hostess'],
            'price': ['price', 'value', 'expensive', 'cheap', 'afford', 'worth', 'cost', 'money', 'bill', 'overpriced'],
            'ambiance': ['ambiance', 'atmosphere', 'decor', 'noise', 'comfortable', 'setting', 'music', 'loud', 'quiet', 'casual', 'classy', 'divey', 'hipster', 'intimate', 
                'romantic', 'touristy', 'trendy', 'upscale']
        }

        # For each aspect, extract relevant sentences and calculate sentiment
        for aspect, keywords in aspects.items():
            # Create regex pattern for the aspect - word boundaries ensure we match whole words
            pattern = '|'.join(r'\b{}\b'.format(word) for word in keywords)
            
            # Extract sentences containing the aspect using the sentiment-preserved text
            def extract_aspect_sentences(text):
                if not text or not isinstance(text, str):
                    return ""
                sentences = [s.strip() for s in text.split('.') if s.strip()]
                aspect_sentences = [s for s in sentences if any(word in s.lower().split() for word in keywords)]
                return ' '.join(aspect_sentences) if aspect_sentences else ''
            
            # Extract sentences related to each aspect
            self.reviews_df[f'{aspect}_text'] = self.reviews_df['sentiment_text'].apply(extract_aspect_sentences)
            
            # Calculate sentiment for those sentences (if they exist)
            def get_aspect_sentiment(text):
                if text:
                    return self.sia.polarity_scores(text)['compound']
                return 0  # Default to neutral if no aspect mentioned
            
            self.reviews_df[f'{aspect}_sentiment'] = self.reviews_df[f'{aspect}_text'].apply(get_aspect_sentiment)
        
        return self
    
    def add_text_complexity_features(self):
        """Add features related to text complexity which can help with fake review detection"""
        # Original text for these metrics to preserve natural writing style
        original_text = self.reviews_df['text']
        
        # Calculate average word length
        self.reviews_df['avg_word_length'] = original_text.apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0
        )
        
        # Calculate sentence count
        self.reviews_df['sentence_count'] = original_text.apply(
            lambda x: len([s for s in str(x).split('.') if s.strip()]) if x else 0
        )
        
        # Calculate average sentence length (words per sentence)
        def avg_sentence_length(text):
            if not text or not isinstance(text, str):
                return 0
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if not sentences:
                return 0
            words_per_sentence = [len(s.split()) for s in sentences]
            return np.mean(words_per_sentence) if words_per_sentence else 0
            
        self.reviews_df['avg_sentence_length'] = original_text.apply(avg_sentence_length)
        
        # Lexical diversity (unique words / total words)
        def lexical_diversity(text):
            if not text or not isinstance(text, str):
                return 0
            words = text.lower().split()
            if not words:
                return 0
            return len(set(words)) / len(words)
            
        self.reviews_df['lexical_diversity'] = original_text.apply(lexical_diversity)
        
        return self
    
    def add_topic_modeling(self, n_topics=5, max_features=1000):
        """Extract topics from reviews using LDA on cleaned text"""
        # Use the fully cleaned text for topic modeling
        clean_texts = self.reviews_df['clean_text'].tolist()
        
        # Create document-term matrix
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=max_features)
        X = vectorizer.fit_transform(clean_texts)
        
        # Fit LDA model
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
        lda.fit(X)
        
        # Transform documents to topic space
        topic_distribution = lda.transform(X)
        
        # Add topic distributions as features
        for i in range(n_topics):
            self.reviews_df[f'topic_{i}'] = topic_distribution[:, i]
        
        # Print top words per topic (for interpretation)
        feature_names = vectorizer.get_feature_names_out()
        top_words_per_topic = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-10 - 1:-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_words_per_topic.append(top_words)
            print(f"Topic {topic_idx}: {', '.join(top_words)}")
        
        return self
    
    def add_user_stats(self):
        """Add user-level statistics as features"""
        # Calculate average rating per user
        user_avg_rating = self.reviews_df.groupby('user_id')['rating'].mean()
        self.reviews_df['user_avg_rating'] = self.reviews_df['user_id'].map(user_avg_rating)
        
        # Calculate rating deviation (how this rating differs from user's average)
        self.reviews_df['rating_deviation'] = self.reviews_df['rating'] - self.reviews_df['user_avg_rating']
        
        # Count reviews per user
        user_review_count = self.reviews_df.groupby('user_id')['review_id'].count()
        self.reviews_df['user_review_count'] = self.reviews_df['user_id'].map(user_review_count)
        
        # Calculate standard deviation of user ratings (consistent vs variable rater)
        user_rating_std = self.reviews_df.groupby('user_id')['rating'].std()
        self.reviews_df['user_rating_std'] = self.reviews_df['user_id'].map(user_rating_std)
        
        return self
    
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
        
        # Handle missing values ?? why zero makes any sense claude ? 
        edge_features = edge_features.fillna(0)
        
        return edge_features

# Example usage:
# from text_preprocessing import TextPreprocessor
# 
# # First parse your JSON data into a DataFrame
# reviews_df = flatten_restaurant_reviews(json_data)
#
# # Then create the enhanced feature extractor
# extractor = EnhancedReviewFeatureExtractor(reviews_df)
# 
# # Apply all feature extraction steps
# edge_features = (extractor
#                 .add_sentiment_scores()
#                 .extract_aspect_sentiments()
#                 .add_text_complexity_features()
#                 .add_topic_modeling()
#                 .add_user_stats()
#                 .build_edge_features())
# 
# print(edge_features.head())