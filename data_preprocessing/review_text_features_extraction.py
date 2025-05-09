import pandas as pd
import numpy as np
from transformers import pipeline
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.aspects import aspects
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Import text preprocessor
from text_preprocessing import TextPreprocessor

class TextFeatureExtractor:
    def __init__(self, reviews_df):
        """Initialize with the flattened dataframe of reviews"""
        self.reviews_df = reviews_df
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()
        # Preprocess the text
        self.reviews_df = self.preprocessor.process_reviews_df(self.reviews_df)
        
        # Initialize sentiment analyzer
        # sia does not work
        self.sia = SentimentIntensityAnalyzer()
        
        # # Initialize BERT for review embeddings
        sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
    
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
        
        return self.reviews_df
    
    def extract_aspect_sentiments(self):
        """Extract sentiment for specific aspects of restaurant reviews"""
       
        aspects = aspects()

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
        
        return self.reviews_df
    
   
    
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
        
        return self.reviews_df
    
