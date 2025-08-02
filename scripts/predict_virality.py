import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils.feature_engineering import create_virality_features
from utils.constants import MODEL_PATHS, VIRALITY_THRESHOLD

class ViralityPredictor:
    """
    Virality prediction model for YouTube videos.
    """
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.is_trained = False
        
    def train(self, data):
        """
        Train the virality prediction model.
        
        Args:
            data (pd.DataFrame): Training data with video metadata
        """
        # Create features
        features_list = []
        labels = []
        
        for _, row in data.iterrows():
            video_data = row.to_dict()
            features = create_virality_features(video_data)
            features_list.append(features)
            
            # Create binary label based on views threshold
            views = video_data.get('views', 0)
            # Use is_viral if available, otherwise calculate
            if 'is_viral' in video_data:
                labels.append(video_data['is_viral'])
            else:
                labels.append(1 if views >= VIRALITY_THRESHOLD else 0)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        self.feature_columns = features_df.columns.tolist()
        
        # Remove rows with NaN values
        valid_indices = ~(features_df.isna().any(axis=1) | pd.isna(labels))
        features_df = features_df[valid_indices]
        labels = [labels[i] for i in range(len(labels)) if valid_indices[i]]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Virality model accuracy: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
    def predict(self, video_data):
        """
        Predict virality probability for a video.
        
        Args:
            video_data (dict): Video metadata
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            return {
                'viral_probability': 0.5,
                'prediction': False,
                'confidence': 0.0
            }
        
        # Create features
        features = create_virality_features(video_data)
        
        # Convert to feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0))
        
        X = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        probability = self.model.predict_proba(X)[0][1]
        prediction = probability > 0.5
        
        # Calculate confidence based on probability
        confidence = abs(probability - 0.5) * 2
        
        return {
            'viral_probability': probability,
            'prediction': prediction,
            'confidence': confidence,
            'threshold': VIRALITY_THRESHOLD
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
            filepath = MODEL_PATHS['virality']
        
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
            filepath = MODEL_PATHS['virality']
        
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

def predict_virality(video_data):
    """
    Predict virality for a video.
    
    Args:
        video_data (dict): Video metadata
        
    Returns:
        dict: Virality prediction results
    """
    predictor = ViralityPredictor()
    predictor.load_model()
    
    return predictor.predict(video_data) 