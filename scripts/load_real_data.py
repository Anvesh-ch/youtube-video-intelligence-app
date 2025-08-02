#!/usr/bin/env python3
"""
Load and preprocess real YouTube dataset from archive folder.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.text_cleaning import clean_text, extract_title_features, extract_description_features
from utils.constants import YOUTUBE_CATEGORIES, VIRALITY_THRESHOLD

def load_category_mappings():
    """Load category ID mappings from JSON files."""
    category_mappings = {}
    
    # Load US categories as default
    us_categories_file = 'archive/US_category_id.json'
    if os.path.exists(us_categories_file):
        with open(us_categories_file, 'r') as f:
            us_categories = json.load(f)
            for item in us_categories['items']:
                category_id = int(item['id'])
                category_name = item['snippet']['title']
                category_mappings[category_id] = category_name
    
    return category_mappings

def load_youtube_data(country='US', max_samples=None):
    """
    Load YouTube trending data for a specific country.
    
    Args:
        country (str): Country code (US, GB, CA, etc.)
        max_samples (int): Maximum number of samples to load
        
    Returns:
        pd.DataFrame: Processed YouTube data
    """
    csv_file = f'archive/{country}videos.csv'
    
    if not os.path.exists(csv_file):
        print(f"Data file not found: {csv_file}")
        return None
    
    print(f"Loading {country} YouTube data...")
    
    # Load the CSV file with proper encoding
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(csv_file, encoding='latin-1')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='cp1252')
    
    print(f"Loaded {len(df)} videos from {country}")
    
    # Limit samples if specified
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
        print(f"Sampled {len(df)} videos")
    
    return df

def preprocess_youtube_data(df, category_mappings):
    """
    Preprocess YouTube data for machine learning.
    
    Args:
        df (pd.DataFrame): Raw YouTube data
        category_mappings (dict): Category ID to name mappings
        
    Returns:
        pd.DataFrame: Processed data
    """
    print("Preprocessing YouTube data...")
    
    # Create a copy to avoid modifying original
    data = df.copy()
    
    # Handle missing values
    data['views'] = data['views'].fillna(0)
    data['likes'] = data['likes'].fillna(0)
    data['comment_count'] = data['comment_count'].fillna(0)
    data['title'] = data['title'].fillna('')
    data['description'] = data['description'].fillna('')
    data['tags'] = data['tags'].fillna('')
    
    # Convert to numeric
    data['views'] = pd.to_numeric(data['views'], errors='coerce').fillna(0)
    data['likes'] = pd.to_numeric(data['likes'], errors='coerce').fillna(0)
    data['comment_count'] = pd.to_numeric(data['comment_count'], errors='coerce').fillna(0)
    
    # Map category IDs to names
    data['category_name'] = data['category_id'].map(category_mappings)
    data['category_name'] = data['category_name'].fillna('Other')
    
    # Process tags
    data['tags_list'] = data['tags'].apply(lambda x: 
        [tag.strip() for tag in str(x).split('|') if tag.strip()] if pd.notna(x) and str(x) != 'nan' else [])
    
    # Extract features
    data['title_length'] = data['title'].str.len()
    data['description_length'] = data['description'].str.len()
    data['tags_count'] = data['tags_list'].str.len()
    
    # Create engagement features
    data['like_ratio'] = data['likes'] / data['views'].replace(0, 1)
    data['comment_ratio'] = data['comment_count'] / data['views'].replace(0, 1)
    data['engagement_rate'] = (data['likes'] + data['comment_count']) / data['views'].replace(0, 1)
    
    # Create virality label
    data['is_viral'] = (data['views'] >= VIRALITY_THRESHOLD).astype(int)
    
    # Process publish time
    data['publish_time'] = pd.to_datetime(data['publish_time'])
    data['publish_hour'] = data['publish_time'].dt.hour
    data['publish_day'] = data['publish_time'].dt.dayofweek
    data['publish_month'] = data['publish_time'].dt.month
    data['is_weekend'] = data['publish_day'].isin([5, 6]).astype(int)
    data['is_business_hours'] = ((data['publish_hour'] >= 9) & (data['publish_hour'] <= 17)).astype(int)
    
    # Extract title features
    title_features = data['title'].apply(extract_title_features)
    title_df = pd.DataFrame(title_features.tolist())
    data = pd.concat([data, title_df], axis=1)
    
    # Extract description features
    desc_features = data['description'].apply(extract_description_features)
    desc_df = pd.DataFrame(desc_features.tolist())
    data = pd.concat([data, desc_df], axis=1)
    
    print(f"Preprocessed {len(data)} videos")
    return data

def create_training_dataset():
    """
    Create a comprehensive training dataset from all available countries.
    
    Returns:
        pd.DataFrame: Combined training dataset
    """
    print("Creating comprehensive training dataset...")
    
    # Load category mappings
    category_mappings = load_category_mappings()
    
    # Countries to process
    countries = ['US', 'GB', 'CA', 'DE', 'FR', 'IN', 'JP', 'KR', 'MX', 'RU']
    
    all_data = []
    
    for country in countries:
        try:
            # Load data for each country
            df = load_youtube_data(country, max_samples=5000)  # Limit per country
            
            if df is not None and len(df) > 0:
                # Preprocess the data
                processed_df = preprocess_youtube_data(df, category_mappings)
                processed_df['country'] = country
                all_data.append(processed_df)
                print(f"Added {len(processed_df)} videos from {country}")
        except Exception as e:
            print(f"Error processing {country}: {e}")
            continue
    
    if not all_data:
        print("No data could be loaded")
        return None
    
    # Combine all datasets
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"Combined dataset: {len(combined_data)} videos")
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    combined_data.to_csv('data/processed_youtube_data.csv', index=False)
    print("Saved processed data to data/processed_youtube_data.csv")
    
    return combined_data

def analyze_dataset(data):
    """
    Analyze the dataset and print statistics.
    
    Args:
        data (pd.DataFrame): Processed dataset
    """
    print("\nDataset Analysis:")
    print("=" * 50)
    print(f"Total videos: {len(data)}")
    print(f"Viral videos: {data['is_viral'].sum()} ({data['is_viral'].mean():.2%})")
    print(f"Average views: {data['views'].mean():,.0f}")
    print(f"Average likes: {data['likes'].mean():,.0f}")
    print(f"Average comments: {data['comment_count'].mean():,.0f}")
    
    print("\nTop categories:")
    category_counts = data['category_name'].value_counts().head(10)
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    print("\nEngagement statistics:")
    print(f"  Average like ratio: {data['like_ratio'].mean():.4f}")
    print(f"  Average comment ratio: {data['comment_ratio'].mean():.4f}")
    print(f"  Average engagement rate: {data['engagement_rate'].mean():.4f}")

if __name__ == "__main__":
    print("YouTube Video Intelligence App - Data Loading")
    print("=" * 50)
    
    # Create training dataset
    data = create_training_dataset()
    
    if data is not None:
        # Analyze the dataset
        analyze_dataset(data)
        
        print("\nData loading completed successfully!")
        print("Next step: Run train_models.py to train the ML models")
    else:
        print("Failed to load data. Please check the archive folder.") 