import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from utils.constants import STOP_WORDS

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text, remove_stopwords=True, max_length=None):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Input text to clean
        remove_stopwords (bool): Whether to remove stop words
        max_length (int): Maximum length to truncate text
    
    Returns:
        str: Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text.strip())
    
    # Remove stop words if requested
    if remove_stopwords:
        stop_words = set(stopwords.words('english') + STOP_WORDS)
        tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Join tokens back
    cleaned_text = ' '.join(tokens)
    
    # Truncate if max_length is specified
    if max_length and len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]
    
    return cleaned_text.strip()

def extract_title_features(title):
    """
    Extract features from video title.
    
    Args:
        title (str): Video title
    
    Returns:
        dict: Dictionary of title features
    """
    if not title or pd.isna(title) or str(title) == 'nan':
        return {
            'title_length': 0,
            'word_count': 0,
            'has_question': False,
            'has_exclamation': False,
            'has_numbers': False,
            'has_uppercase': False,
            'sentiment_score': 0.0
        }
    
    # Basic features
    features = {
        'title_length': len(title),
        'word_count': len(title.split()),
        'has_question': '?' in title,
        'has_exclamation': '!' in title,
        'has_numbers': bool(re.search(r'\d', title)),
        'has_uppercase': any(c.isupper() for c in title),
        'sentiment_score': TextBlob(title).sentiment.polarity
    }
    
    return features

def extract_description_features(description):
    """
    Extract features from video description.
    
    Args:
        description (str): Video description
    
    Returns:
        dict: Dictionary of description features
    """
    if not description or pd.isna(description) or str(description) == 'nan':
        return {
            'description_length': 0,
            'word_count': 0,
            'has_links': False,
            'has_hashtags': False,
            'sentiment_score': 0.0
        }
    
    # Basic features
    features = {
        'description_length': len(description),
        'word_count': len(description.split()),
        'has_links': bool(re.search(r'http[s]?://', description)),
        'has_hashtags': '#' in description,
        'sentiment_score': TextBlob(description).sentiment.polarity
    }
    
    return features

def process_tags(tags):
    """
    Process and clean video tags.
    
    Args:
        tags (list): List of video tags
    
    Returns:
        list: Cleaned tags list
    """
    if not tags or pd.isna(tags) or str(tags) == 'nan':
        return []
    
    # Handle case where tags is a string
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split('|') if tag.strip()]
    
    # Handle case where tags is not a list
    if not isinstance(tags, list):
        return []
    
    cleaned_tags = []
    for tag in tags:
        if tag and isinstance(tag, str):
            cleaned_tag = clean_text(tag, remove_stopwords=False)
            if cleaned_tag and len(cleaned_tag) > 2:
                cleaned_tags.append(cleaned_tag)
    
    return cleaned_tags[:50]  # Limit to 50 tags

def extract_publish_time_features(publish_time):
    """
    Extract features from publish time.
    
    Args:
        publish_time (str): ISO format publish time
    
    Returns:
        dict: Dictionary of time features
    """
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(publish_time.replace('Z', '+00:00'))
        
        return {
            'publish_hour': dt.hour,
            'publish_day': dt.weekday(),
            'publish_month': dt.month,
            'is_weekend': dt.weekday() >= 5,
            'is_business_hours': 9 <= dt.hour <= 17
        }
    except:
        return {
            'publish_hour': 12,
            'publish_day': 0,
            'publish_month': 1,
            'is_weekend': False,
            'is_business_hours': True
        } 