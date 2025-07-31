import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.text_cleaning import clean_text, process_tags
from utils.constants import MODEL_PATHS

class TagRecommender:
    """
    Tag recommendation model for YouTube videos.
    """
    
    def __init__(self):
        self.model = None
        self.tag_embeddings = None
        self.tag_list = None
        self.is_trained = False
        
    def train(self, data):
        """
        Train the tag recommendation model.
        
        Args:
            data (pd.DataFrame): Training data with video metadata
        """
        # Collect all unique tags
        all_tags = set()
        for _, row in data.iterrows():
            tags = row.get('tags', [])
            if isinstance(tags, list):
                processed_tags = process_tags(tags)
                all_tags.update(processed_tags)
        
        self.tag_list = list(all_tags)
        
        # Initialize sentence transformer
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create embeddings for all tags
        tag_texts = [f"tag: {tag}" for tag in self.tag_list]
        self.tag_embeddings = self.model.encode(tag_texts)
        
        self.is_trained = True
        print(f"Trained tag recommender with {len(self.tag_list)} tags")
        
    def recommend_tags(self, video_data, top_k=5):
        """
        Recommend tags for a video.
        
        Args:
            video_data (dict): Video metadata
            top_k (int): Number of tags to recommend
            
        Returns:
            list: Recommended tags
        """
        if not self.is_trained:
            return []
        
        # Create query text from title and category
        title = video_data.get('title', '')
        category = video_data.get('category', '')
        
        query_text = f"title: {title} category: {category}"
        
        # Get query embedding
        query_embedding = self.model.encode([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.tag_embeddings)[0]
        
        # Get top-k tags
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommended_tags = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                recommended_tags.append({
                    'tag': self.tag_list[idx],
                    'similarity': similarities[idx]
                })
        
        return recommended_tags
    
    def save_model(self, filepath=None):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save model
        """
        if not self.is_trained:
            print("Model not trained yet")
            return
        
        if filepath is None:
            filepath = MODEL_PATHS['tag_embeddings']
        
        model_data = {
            'model': self.model,
            'tag_embeddings': self.tag_embeddings,
            'tag_list': self.tag_list,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load a trained model.
        
        Args:
            filepath (str): Path to load model from
        """
        if filepath is None:
            filepath = MODEL_PATHS['tag_embeddings']
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.tag_embeddings = model_data['tag_embeddings']
            self.tag_list = model_data['tag_list']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            self.is_trained = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False

def recommend_tags(video_data, top_k=5):
    """
    Recommend tags for a video.
    
    Args:
        video_data (dict): Video metadata
        top_k (int): Number of tags to recommend
        
    Returns:
        list: Recommended tags with similarity scores
    """
    recommender = TagRecommender()
    recommender.load_model()
    
    return recommender.recommend_tags(video_data, top_k) 