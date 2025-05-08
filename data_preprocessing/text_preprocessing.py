import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    """Text preprocessing class for cleaning and normalizing review text"""
    
    def __init__(self):
        """Initialize the preprocessor with required NLTK resources"""
        # Download necessary NLTK resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        # Get English stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def _remove_urls(self, text):
        """Remove URLs from text"""
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    
    def _remove_html_tags(self, text):
        """Remove HTML tags from text"""
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)
    
    def _remove_emoji(self, text):
        """Remove emojis from text"""
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    def _standardize_text(self, text):
        """Standardize text by converting to lowercase, removing extra spaces"""
        # Convert to lowercase
        text = text.lower()
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove leading and trailing spaces
        text = text.strip()
        return text
    
    def _remove_punctuation(self, text):
        """Remove punctuation from text"""
        translator = str.maketrans('', '', string.punctuation)
        return text.translate(translator)
    
    def _remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def _lemmatize_text(self, text):
        """Lemmatize text - convert words to their base form"""
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def _remove_numeric(self, text):
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)
    
    def clean_text(self, text, for_sentiment=False):
        """
        Clean text with all preprocessing steps
        
        Args:
            text (str): Text to clean
            for_sentiment (bool): If True, preserves some elements important for sentiment analysis
        
        Returns:
            str: Cleaned text
        """
        if text is None or not isinstance(text, str):
            return ""
            
        # Remove URLs, HTML tags, and emojis
        text = self._remove_urls(text)
        text = self._remove_html_tags(text)
        text = self._remove_emoji(text)
        
        # Standardize text
        text = self._standardize_text(text)
        
        if for_sentiment:
            # For sentiment analysis, we want to preserve more of the original text
            # Just remove extra spaces and standardize
            return text
        else:
            # For other NLP tasks like topic modeling, do more aggressive cleaning
            text = self._remove_punctuation(text)
            text = self._remove_numeric(text)
            text = self._remove_stopwords(text)
            text = self._lemmatize_text(text)
            return text
    
    def process_reviews_df(self, df, text_column='text'):
        """
        Process all reviews in a dataframe
        
        Args:
            df (pd.DataFrame): DataFrame containing reviews
            text_column (str): Name of column containing review text
            
        Returns:
            pd.DataFrame: DataFrame with additional cleaned text columns
        """
        # Make a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Add cleaned text columns
        processed_df['clean_text'] = processed_df[text_column].apply(
            lambda x: self.clean_text(x, for_sentiment=False)
        )
        
        processed_df['sentiment_text'] = processed_df[text_column].apply(
            lambda x: self.clean_text(x, for_sentiment=True)
        )
        
        return processed_df

# Example usage:
# preprocessor = TextPreprocessor()
# cleaned_df = preprocessor.process_reviews_df(reviews_df)
# 
# # Now use the cleaned text for your NLP tasks
# # For sentiment analysis:
# sia = SentimentIntensityAnalyzer()
# cleaned_df['sentiment'] = cleaned_df['sentiment_text'].apply(lambda x: sia.polarity_scores(x)['compound'])
# 
# # For topic modeling:
# vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
# X = vectorizer.fit_transform(cleaned_df['clean_text'])