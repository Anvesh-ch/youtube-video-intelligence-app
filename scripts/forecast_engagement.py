import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from utils.feature_engineering import create_engagement_features
from utils.constants import MODEL_PATHS

class EngagementForecaster:
    """
    Engagement forecasting model for YouTube videos.
    """
    
    def __init__(self):
        self.views_model = None
        self.likes_model = None
        self.comments_model = None
        self.feature_columns = None
        self.is_trained = False
        
    def train(self, data):
        """
        Train the engagement forecasting models.
        
        Args:
            data (pd.DataFrame): Training data with video metadata
        """
        # Create features
        features_list = []
        views_labels = []
        likes_labels = []
        comments_labels = []
        
        for _, row in data.iterrows():
            video_data = row.to_dict()
            features = create_engagement_features(video_data)
            features_list.append(features)
            
            # Extract labels
            views_labels.append(video_data.get('views', 0))
            likes_labels.append(video_data.get('likes', 0))
            comments_labels.append(video_data.get('comments', 0))
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_columns = features_df.columns.tolist()
        
        # Split data
        X_train, X_test, y_views_train, y_views_test = train_test_split(
            features_df, views_labels, test_size=0.2, random_state=42
        )
        _, _, y_likes_train, y_likes_test = train_test_split(
            features_df, likes_labels, test_size=0.2, random_state=42
        )
        _, _, y_comments_train, y_comments_test = train_test_split(
            features_df, comments_labels, test_size=0.2, random_state=42
        )
        
        # Train views model
        self.views_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.views_model.fit(X_train, y_views_train)
        
        # Train likes model
        self.likes_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.likes_model.fit(X_train, y_likes_train)
        
        # Train comments model
        self.comments_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.comments_model.fit(X_train, y_comments_train)
        
        # Evaluate models
        y_views_pred = self.views_model.predict(X_test)
        y_likes_pred = self.likes_model.predict(X_test)
        y_comments_pred = self.comments_model.predict(X_test)
        
        print(f"Views model R2: {r2_score(y_views_test, y_views_pred):.3f}")
        print(f"Likes model R2: {r2_score(y_likes_test, y_likes_pred):.3f}")
        print(f"Comments model R2: {r2_score(y_comments_test, y_comments_pred):.3f}")
        
        self.is_trained = True
        
    def predict(self, video_data):
        """
        Predict engagement metrics for a video.
        
        Args:
            video_data (dict): Video metadata
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            return {
                'predicted_views': 0,
                'predicted_likes': 0,
                'predicted_comments': 0,
                'confidence': 0.0
            }
        
        # Create features
        features = create_engagement_features(video_data)
        
        # Convert to feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make predictions
        predicted_views = max(0, int(self.views_model.predict(X)[0]))
        predicted_likes = max(0, int(self.likes_model.predict(X)[0]))
        predicted_comments = max(0, int(self.comments_model.predict(X)[0]))
        
        # Calculate confidence (average of model feature importances)
        confidence = 0.7  # Placeholder
        
        return {
            'predicted_views': predicted_views,
            'predicted_likes': predicted_likes,
            'predicted_comments': predicted_comments,
            'confidence': confidence,
            'engagement_rate': (predicted_likes + predicted_comments) / max(predicted_views, 1)
        }
    
    def save_model(self, filepath=None):
        """
        Save the trained models.
        
        Args:
            filepath (str): Path to save models
        """
        if not self.is_trained:
            print("Models not trained yet")
            return
        
        if filepath is None:
            filepath = MODEL_PATHS['engagement']
        
        model_data = {
            'views_model': self.views_model,
            'likes_model': self.likes_model,
            'comments_model': self.comments_model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")
    
    def load_model(self, filepath=None):
        """
        Load trained models.
        
        Args:
            filepath (str): Path to load models from
        """
        if filepath is None:
            filepath = MODEL_PATHS['engagement']
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.views_model = model_data['views_model']
            self.likes_model = model_data['likes_model']
            self.comments_model = model_data['comments_model']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            print(f"Models loaded from {filepath}")
            
        except FileNotFoundError:
            print(f"Model file not found: {filepath}")
            self.is_trained = False
        except Exception as e:
            print(f"Error loading models: {e}")
            self.is_trained = False

def forecast_engagement(video_data):
    """
    Forecast engagement metrics for a video.
    
    Args:
        video_data (dict): Video metadata
        
    Returns:
        dict: Engagement forecast results
    """
    forecaster = EngagementForecaster()
    forecaster.load_model()
    
    return forecaster.predict(video_data) 