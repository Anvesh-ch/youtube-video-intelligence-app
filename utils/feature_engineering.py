import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.text_cleaning import (
    clean_text, extract_title_features, extract_description_features,
    process_tags, extract_publish_time_features
)
from utils.constants import YOUTUBE_CATEGORIES, DEFAULT_VALUES

class FeatureEngineer:
    """
    Feature engineering class for YouTube video analysis.
    """
    
    def __init__(self):
        self.title_vectorizer = None
        self.category_encoder = None
        self.scaler = None
        self.feature_columns = []
        
    def fit(self, data):
        """
        Fit the feature engineering pipeline on training data.
        
        Args:
            data (pd.DataFrame): Training data with video metadata
        """
        # Initialize vectorizers and encoders
        self.title_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        self.category_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Fit title vectorizer
        titles = data['title'].fillna('')
        self.title_vectorizer.fit(titles)
        
        # Fit category encoder
        categories = data['category'].fillna('Other')
        self.category_encoder.fit(categories)
        
        # Define feature columns
        self.feature_columns = [
            'title_length', 'title_word_count', 'title_has_question',
            'title_has_exclamation', 'title_has_numbers', 'title_has_uppercase',
            'title_sentiment_score', 'description_length', 'description_word_count',
            'description_has_links', 'description_has_hashtags', 'description_sentiment_score',
            'tag_count', 'publish_hour', 'publish_day', 'publish_month',
            'is_weekend', 'is_business_hours', 'category_encoded'
        ]
        
    def transform(self, video_data):
        """
        Transform video metadata into features.
        
        Args:
            video_data (dict): Video metadata dictionary
            
        Returns:
            np.ndarray: Feature vector
        """
        # Extract basic features
        features = {}
        
        # Title features
        title = video_data.get('title', DEFAULT_VALUES['title'])
        title_features = extract_title_features(title)
        features.update({
            'title_length': title_features['title_length'],
            'title_word_count': title_features['word_count'],
            'title_has_question': title_features['has_question'],
            'title_has_exclamation': title_features['has_exclamation'],
            'title_has_numbers': title_features['has_numbers'],
            'title_has_uppercase': title_features['has_uppercase'],
            'title_sentiment_score': title_features['sentiment_score']
        })
        
        # Description features
        description = video_data.get('description', DEFAULT_VALUES['description'])
        desc_features = extract_description_features(description)
        features.update({
            'description_length': desc_features['description_length'],
            'description_word_count': desc_features['word_count'],
            'description_has_links': desc_features['has_links'],
            'description_has_hashtags': desc_features['has_hashtags'],
            'description_sentiment_score': desc_features['sentiment_score']
        })
        
        # Tag features
        tags = video_data.get('tags', DEFAULT_VALUES['tags'])
        processed_tags = process_tags(tags)
        features['tag_count'] = len(processed_tags)
        
        # Publish time features
        publish_time = video_data.get('publish_time', '')
        time_features = extract_publish_time_features(publish_time)
        features.update({
            'publish_hour': time_features['publish_hour'],
            'publish_day': time_features['publish_day'],
            'publish_month': time_features['publish_month'],
            'is_weekend': time_features['is_weekend'],
            'is_business_hours': time_features['is_business_hours']
        })
        
        # Category encoding
        category = video_data.get('category', DEFAULT_VALUES['category'])
        try:
            category_encoded = self.category_encoder.transform([category])[0]
        except:
            category_encoded = 0
        features['category_encoded'] = category_encoded
        
        # Convert to feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        return np.array(feature_vector).reshape(1, -1)
    
    def get_title_embedding(self, title):
        """
        Get TF-IDF embedding for title.
        
        Args:
            title (str): Video title
            
        Returns:
            np.ndarray: Title embedding
        """
        if self.title_vectorizer is None:
            return np.zeros((1, 100))
        
        title_clean = clean_text(title, max_length=100)
        return self.title_vectorizer.transform([title_clean]).toarray()

def create_engagement_features(video_data):
    """
    Create features for engagement prediction.
    
    Args:
        video_data (dict): Video metadata
        
    Returns:
        dict: Engagement features
    """
    features = {}
    
    # Basic engagement metrics
    views = video_data.get('views', 0)
    likes = video_data.get('likes', 0)
    comments = video_data.get('comment_count', video_data.get('comments', 0))
    
    features.update({
        'views': views,
        'likes': likes,
        'comments': comments,
        'like_ratio': likes / max(views, 1),
        'comment_ratio': comments / max(views, 1),
        'engagement_rate': (likes + comments) / max(views, 1)
    })
    
    # Title and description features
    title = video_data.get('title', '')
    description = video_data.get('description', '')
    
    title_features = extract_title_features(title)
    desc_features = extract_description_features(description)
    
    features.update({
        'title_length': title_features['title_length'],
        'title_word_count': title_features['word_count'],
        'title_sentiment': title_features['sentiment_score'],
        'description_length': desc_features['description_length'],
        'description_word_count': desc_features['word_count'],
        'description_sentiment': desc_features['sentiment_score']
    })
    
    # Tag features
    tags = video_data.get('tags_list', video_data.get('tags', []))
    if isinstance(tags, str):
        tags = [tag.strip() for tag in tags.split('|') if tag.strip()]
    features['tag_count'] = len(process_tags(tags))
    
    # Time features
    publish_time = video_data.get('publish_time', '')
    time_features = extract_publish_time_features(publish_time)
    features.update({
        'publish_hour': time_features['publish_hour'],
        'is_weekend': time_features['is_weekend'],
        'is_business_hours': time_features['is_business_hours']
    })
    
    return features

def create_virality_features(video_data):
    """
    Create features for virality prediction.
    
    Args:
        video_data (dict): Video metadata
        
    Returns:
        dict: Virality features
    """
    features = create_engagement_features(video_data)
    
    # Add virality-specific features
    title = video_data.get('title', '')
    if pd.isna(title) or str(title) == 'nan':
        title = ''
    
    features.update({
        'has_viral_keywords': any(keyword in title.lower() for keyword in 
                                 ['viral', 'trending', 'shocking', 'amazing', 'incredible']),
        'title_has_question': '?' in title,
        'title_has_exclamation': '!' in title,
        'title_has_numbers': any(c.isdigit() for c in title)
    })
    
    return features 