import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.predict_virality import ViralityPredictor
from scripts.predict_title_score import TitleScorePredictor
from scripts.forecast_engagement import EngagementForecaster
from scripts.recommend_tags import TagRecommender
from utils.constants import MODEL_PATHS

def generate_sample_data(n_samples=1000):
    """
    Generate sample data for training models.
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Sample data
    """
    np.random.seed(42)
    random.seed(42)
    
    # Sample titles
    titles = [
        "Amazing Life Hacks You Need to Know",
        "10 Incredible Facts About Space",
        "How to Make the Perfect Pizza",
        "The Most Shocking Moments in History",
        "Tutorial: Learn Python in 10 Minutes",
        "Viral Dance Challenge Compilation",
        "Cooking with Gordon Ramsay",
        "Top 10 Gaming Moments of 2023",
        "Science Experiments You Can Do at Home",
        "Travel Vlog: Exploring Japan"
    ]
    
    # Sample categories
    categories = [
        "Entertainment", "Education", "Gaming", "Howto & Style",
        "Science & Technology", "Music", "Comedy", "News & Politics"
    ]
    
    # Sample channels
    channels = [
        "TechChannel", "CookingMaster", "GamingPro", "ScienceExplained",
        "TravelVlogger", "ComedyCentral", "NewsNetwork", "MusicChannel"
    ]
    
    # Generate data
    data = []
    for i in range(n_samples):
        # Random selection
        title = random.choice(titles) + f" #{i}"
        category = random.choice(categories)
        channel = random.choice(channels)
        
        # Generate realistic metrics
        views = np.random.lognormal(10, 1.5)  # Log-normal distribution
        likes = views * np.random.beta(2, 8)  # Beta distribution for like ratio
        comments = views * np.random.beta(1, 20)  # Beta distribution for comment ratio
        
        # Generate tags
        tag_options = ["tutorial", "viral", "amazing", "incredible", "shocking", 
                      "tutorial", "howto", "tips", "tricks", "guide", "review",
                      "compilation", "best", "top", "ultimate", "complete"]
        tags = random.sample(tag_options, random.randint(3, 8))
        
        # Generate description
        description = f"This is a sample description for {title}. " + \
                     "This video contains amazing content that you won't want to miss. " + \
                     "Subscribe for more content like this!"
        
        # Generate publish time
        publish_time = datetime.now() - timedelta(days=random.randint(1, 365))
        
        data.append({
            'title': title,
            'description': description,
            'channel_title': channel,
            'category': category,
            'tags': tags,
            'views': int(views),
            'likes': int(likes),
            'comments': int(comments),
            'publish_time': publish_time.isoformat(),
            'video_id': f"sample_{i}"
        })
    
    return pd.DataFrame(data)

def train_all_models():
    """
    Train all ML models using sample data.
    """
    print("Generating sample data...")
    data = generate_sample_data(1000)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("Training virality prediction model...")
    virality_predictor = ViralityPredictor()
    virality_predictor.train(data)
    virality_predictor.save_model()
    
    print("Training title quality scoring model...")
    title_predictor = TitleScorePredictor()
    title_predictor.train(data)
    title_predictor.save_model()
    
    print("Training engagement forecasting models...")
    engagement_forecaster = EngagementForecaster()
    engagement_forecaster.train(data)
    engagement_forecaster.save_model()
    
    print("Training tag recommendation model...")
    tag_recommender = TagRecommender()
    tag_recommender.train(data)
    tag_recommender.save_model()
    
    print("All models trained successfully!")
    print(f"Models saved to: {os.path.abspath('models/')}")

def test_models():
    """
    Test the trained models with sample data.
    """
    print("Testing trained models...")
    
    # Sample video data
    sample_video = {
        'title': 'Amazing Life Hacks You Need to Know',
        'description': 'This video contains incredible life hacks that will change your life forever.',
        'channel_title': 'LifeHacksChannel',
        'category': 'Howto & Style',
        'tags': ['lifehacks', 'tips', 'tricks', 'amazing'],
        'views': 50000,
        'likes': 2500,
        'comments': 150,
        'publish_time': '2023-12-01T10:00:00Z'
    }
    
    # Test virality prediction
    from scripts.predict_virality import predict_virality
    virality_result = predict_virality(sample_video)
    print(f"Virality prediction: {virality_result}")
    
    # Test title scoring
    from scripts.predict_title_score import predict_title_score
    title_result = predict_title_score(sample_video)
    print(f"Title score: {title_result}")
    
    # Test engagement forecasting
    from scripts.forecast_engagement import forecast_engagement
    engagement_result = forecast_engagement(sample_video)
    print(f"Engagement forecast: {engagement_result}")
    
    # Test tag recommendations
    from scripts.recommend_tags import recommend_tags
    tag_result = recommend_tags(sample_video)
    print(f"Tag recommendations: {tag_result}")
    
    # Test description summarization
    from scripts.summarize_description import summarize_description
    summary_result = summarize_description(sample_video['description'])
    print(f"Description summary: {summary_result}")

if __name__ == "__main__":
    print("YouTube Video Intelligence App - Model Training")
    print("=" * 50)
    
    train_all_models()
    print("\n" + "=" * 50)
    test_models()
    
    print("\nTraining completed successfully!")
    print("You can now run the Streamlit app with: streamlit run app/app.py") 