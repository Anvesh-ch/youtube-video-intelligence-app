import os
from dotenv import load_dotenv

load_dotenv()

# YouTube API Configuration
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', '')

# Model thresholds and parameters
VIRALITY_THRESHOLD = 100000  # Views threshold for viral classification
VIRALITY_TIME_WINDOW = 48    # Hours to consider for viral prediction

# Feature engineering constants
MAX_TITLE_LENGTH = 100
MAX_DESCRIPTION_LENGTH = 5000
MAX_TAGS_COUNT = 50

# Model file paths
MODEL_PATHS = {
    'virality': 'models/virality_model.pkl',
    'title_score': 'models/title_score_model.pkl',
    'engagement': 'models/engagement_model.pkl',
    'tag_embeddings': 'models/tag_embedding_model.pkl',
    'summary': 'models/t5_summary_model'
}

# YouTube categories mapping
YOUTUBE_CATEGORIES = {
    1: 'Film & Animation',
    2: 'Autos & Vehicles',
    10: 'Music',
    15: 'Pets & Animals',
    17: 'Sports',
    19: 'Travel & Events',
    20: 'Gaming',
    22: 'People & Blogs',
    23: 'Comedy',
    24: 'Entertainment',
    25: 'News & Politics',
    26: 'Howto & Style',
    27: 'Education',
    28: 'Science & Technology',
    29: 'Nonprofits & Activism'
}

# Default values for missing data
DEFAULT_VALUES = {
    'views': 0,
    'likes': 0,
    'comments': 0,
    'title': 'Untitled',
    'description': 'No description available',
    'tags': [],
    'category': 'Other'
}

# Text processing
STOP_WORDS = [
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
] 