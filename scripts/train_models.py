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

def load_real_data():
    """
    Load the real YouTube dataset.
    
    Returns:
        pd.DataFrame: Real YouTube data
    """
    data_file = 'data/processed_youtube_data.csv'
    
    if not os.path.exists(data_file):
        print("Processed data not found. Loading and preprocessing real data...")
        from scripts.load_real_data import create_training_dataset
        data = create_training_dataset()
    else:
        print("Loading processed YouTube data...")
        data = pd.read_csv(data_file)
    
    if data is None or len(data) == 0:
        print("No data available. Please check the archive folder.")
        return None
    
    print(f"Loaded {len(data)} real YouTube videos")
    return data

def train_all_models():
    """
    Train all ML models using real YouTube data.
    """
    print("Loading real YouTube dataset...")
    data = load_real_data()
    
    if data is None:
        print("Failed to load data. Cannot train models.")
        return False
    
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
    return True

def test_models():
    """
    Test the trained models with real YouTube data.
    """
    print("Testing trained models...")
    
    # Sample video data from real dataset
    data_file = 'data/processed_youtube_data.csv'
    if os.path.exists(data_file):
        data = pd.read_csv(data_file)
        sample_video = data.iloc[0].to_dict()
    else:
        # Fallback sample data
        sample_video = {
            'title': 'Amazing Life Hacks You Need to Know',
            'description': 'This video contains incredible life hacks that will change your life forever.',
            'channel_title': 'LifeHacksChannel',
            'category_name': 'Howto & Style',
            'tags_list': ['lifehacks', 'tips', 'tricks', 'amazing'],
            'views': 50000,
            'likes': 2500,
            'comment_count': 150,
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
    
    success = train_all_models()
    
    if success:
        print("\n" + "=" * 50)
        test_models()
        
        print("\nTraining completed successfully!")
        print("You can now run the Streamlit app with: streamlit run app/app.py")
    else:
        print("\nTraining failed. Please check the data and try again.") 