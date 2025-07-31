import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.feature_engineering import extract_title_features
from utils.constants import MODEL_PATHS

class TitleScorePredictor:
    """
    Title quality scoring model for YouTube videos.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def train(self, data):
        """
        Train the title scoring model.
        
        Args:
            data (pd.DataFrame): Training data with video metadata
        """
        # Create features and labels
        features_list = []
        labels = []
        
        for _, row in data.iterrows():
            title = row.get('title', '')
            views = row.get('views', 0)
            likes = row.get('likes', 0)
            
            # Create engagement-based label (normalized score 0-100)
            if views > 0:
                engagement_score = (likes / views) * 1000  # Normalize
                score = min(engagement_score * 10, 100)  # Scale to 0-100
            else:
                score = 0
            
            # Extract title features
            title_features = extract_title_features(title)
            features_list.append(title_features)
            labels.append(score)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_columns = features_df.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Title score model MSE: {mse:.3f}")
        print(f"Title score model R2: {r2:.3f}")
        
        self.is_trained = True
        
    def predict(self, video_data):
        """
        Predict title quality score for a video.
        
        Args:
            video_data (dict): Video metadata
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            return {
                'score': 50.0,
                'confidence': 0.0,
                'feedback': 'Model not trained'
            }
        
        # Extract title features
        title = video_data.get('title', '')
        title_features = extract_title_features(title)
        
        # Convert to feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(title_features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        score = self.model.predict(X)[0]
        score = max(0, min(100, score))  # Clamp to 0-100
        
        # Generate feedback based on score
        if score >= 80:
            feedback = "Excellent title with high engagement potential"
        elif score >= 60:
            feedback = "Good title with moderate engagement potential"
        elif score >= 40:
            feedback = "Average title with room for improvement"
        else:
            feedback = "Title needs improvement for better engagement"
        
        # Calculate confidence based on model's feature importance
        confidence = 0.7  # Placeholder - could be based on prediction variance
        
        return {
            'score': round(score, 1),
            'confidence': confidence,
            'feedback': feedback
        }
    
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
            filepath = MODEL_PATHS['title_score']
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
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
            filepath = MODEL_PATHS['title_score']
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            self.is_trained = False
        except Exception as e:
            print(f"Error loading model: {e}")
            self.is_trained = False

def predict_title_score(video_data):
    """
    Predict title quality score for a video.
    
    Args:
        video_data (dict): Video metadata
        
    Returns:
        dict: Title score prediction results
    """
    predictor = TitleScorePredictor()
    predictor.load_model()
    
    return predictor.predict(video_data) 