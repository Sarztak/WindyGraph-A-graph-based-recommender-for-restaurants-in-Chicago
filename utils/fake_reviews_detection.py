import numpy as np
import pandas as pd

# TBD later into the unknown future; meaning I may not visit this again depending on 
# how much I care about adding this functionality. 

def add_text_complexity_features(reviews_df):
    """Add features related to text complexity which can help with fake review detection"""
    # Original text for these metrics to preserve natural writing style
    original_text = reviews_df['text']
    
    # Calculate average word length
    reviews_df['avg_word_length'] = original_text.apply(
        lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0
    )
    
    # Calculate sentence count
    reviews_df['sentence_count'] = original_text.apply(
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
        
    reviews_df['avg_sentence_length'] = original_text.apply(avg_sentence_length)
    
    # Lexical diversity (unique words / total words)
    def lexical_diversity(text):
        if not text or not isinstance(text, str):
            return 0
        words = text.lower().split()
        if not words:
            return 0
        return len(set(words)) / len(words)
        
    reviews_df['lexical_diversity'] = original_text.apply(lexical_diversity)
    
    return reviews_df