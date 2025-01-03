import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import losses
import logging
import joblib
from typing import List, Tuple, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

class FakeAccountDetector:
    """A comprehensive system for detecting fake social media accounts using hybrid ML models."""
    
    def __init__(self, features: List[str] = None):
        """
        Initialize the detector with specified features.
        
        Args:
            features: List of feature columns to use for detection
        """
        self.features = features or [
            "statuses_count", "followers_count", "friends_count",
            "favourites_count", "listed_count", "default_profile",
            "default_profile_image", "geo_enabled"
        ]
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging for the detector."""
        
        # Create the logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create a file handler to store logs in the 'logs' directory
        log_file_path = os.path.join('logs', 'detector.log')
        
        # Set up the logging configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # For logging to the console
                logging.FileHandler(log_file_path)  # For logging to a file
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Logging is set up.")

    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from existing columns.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        df = data.copy()
        
        # Safe division function
        def safe_divide(a, b):
            return a / (b + 1e-10)
        
        # Basic rate features
        if 'followers_count' in df.columns and 'friends_count' in df.columns:
            df['followers_friends_ratio'] = safe_divide(df['followers_count'], df['friends_count'])
        
        if 'statuses_count' in df.columns:
            # Calculate posting frequency (statuses per follower)
            if 'followers_count' in df.columns:
                df['statuses_per_follower'] = safe_divide(df['statuses_count'], df['followers_count'])
            
            # Calculate engagement rate
            if 'favourites_count' in df.columns:
                df['engagement_rate'] = safe_divide(df['favourites_count'], df['statuses_count'])
        
        # Profile completeness score
        profile_fields = ['default_profile', 'default_profile_image', 'geo_enabled']
        available_fields = [f for f in profile_fields if f in df.columns]
        if available_fields:
            df['profile_completeness'] = df[available_fields].astype(int).mean(axis=1)
        
        # Network reach
        if 'followers_count' in df.columns and 'listed_count' in df.columns:
            df['network_reach'] = df['followers_count'] + df['listed_count']
        
        # If created_at exists, calculate account age
        if 'created_at' in df.columns:
            try:
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['account_age_days'] = (datetime.now() - df['created_at']).dt.days
                if 'statuses_count' in df.columns:
                    df['statuses_per_day'] = safe_divide(df['statuses_count'], df['account_age_days'])
            except:
                self.logger.warning("Could not process created_at column for account age calculation")
        
        return df
    
    def load_and_preprocess_data(self, real_users_path: str, fake_users_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess the dataset.
        
        Args:
            real_users_path: Path to real users CSV
            fake_users_path: Path to fake users CSV
            
        Returns:
            Tuple of features (X) and labels (y)
        """
        self.logger.info("Loading datasets...")
        users = pd.read_csv(real_users_path)
        fusers = pd.read_csv(fake_users_path)
        
        users['is_fake'] = 0
        fusers['is_fake'] = 1
        
        data = pd.concat([users, fusers], ignore_index=True)
        
        # Handle missing values
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna("Unknown")
            else:
                data[col] = data[col].fillna(data[col].median())
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Update features list with any new engineered features that exist in the data
        available_features = [f for f in self.features if f in data.columns]
        engineered_features = [
            'followers_friends_ratio', 'statuses_per_follower', 'engagement_rate',
            'profile_completeness', 'network_reach', 'statuses_per_day'
        ]
        available_engineered = [f for f in engineered_features if f in data.columns]
        
        self.features = available_features + available_engineered
        self.logger.info(f"Using features: {self.features}")
        
        X = data[self.features]
        y = data['is_fake']
        
        # Encode categorical variables
        for col in X.select_dtypes(include=["object", "category"]).columns:
            self.encoders[col] = LabelEncoder()
            X[col] = self.encoders[col].fit_transform(X[col].astype(str))
        
        return X, y
    
    def prepare_lstm_data(self, X: np.ndarray) -> np.ndarray:
        """Reshape data for LSTM input."""
        return np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Sequential:
        """
        Build and compile LSTM model.
        
        Args:
            input_shape: Shape of input data
            
        Returns:
            Compiled LSTM model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=64, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(units=32, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss=losses.BinaryCrossentropy(),
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train all models and return performance metrics.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of model performances
        """
        self.logger.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.scalers['standard'] = MinMaxScaler()
        X_train_scaled = self.scalers['standard'].fit_transform(X_train)
        X_test_scaled = self.scalers['standard'].transform(X_test)
        
        # Train base models
        self.models['logistic'] = LogisticRegression()
        self.models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            if name == 'logistic':
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)
        
        # Train voting classifier
        self.models['voting'] = VotingClassifier(estimators=[ 
            ('lr', self.models['logistic']),
            ('rf', self.models['rf']),
            ('gb', self.models['gb'])
        ], voting='soft')
        
        self.models['voting'].fit(X_train_scaled, y_train)
        
        # Train LSTM
        X_train_lstm = self.prepare_lstm_data(X_train_scaled)
        X_test_lstm = self.prepare_lstm_data(X_test_scaled)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        self.models['lstm'] = self.build_lstm_model((X_train_lstm.shape[1], 1))
        self.models['lstm'].fit(
            X_train_lstm,
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate all models
        results = {}
        for name, model in self.models.items():
            if name == 'lstm':
                y_pred = (model.predict(X_test_lstm) > 0.5).astype(int)
            elif name in ['logistic', 'voting']:
                y_pred = model.predict(X_test_scaled)
            else:
                y_pred = model.predict(X_test)
            
            results[f'{name}_accuracy'] = accuracy_score(y_test, y_pred)
            results[f'{name}_auc'] = roc_auc_score(y_test, y_pred)
        
        # Create ensemble predictions
        voting_pred = self.models['voting'].predict_proba(X_test_scaled)[:, 1]
        lstm_pred = self.models['lstm'].predict(X_test_lstm).flatten()
        ensemble_pred = (0.6 * voting_pred + 0.4 * lstm_pred > 0.5).astype(int)
        
        results['ensemble_accuracy'] = accuracy_score(y_test, ensemble_pred)
        results['ensemble_auc'] = roc_auc_score(y_test, ensemble_pred)
        
        self.logger.info("Training completed successfully")
        return results
    
    def save_models(self, path: str):
        """Save all models and preprocessors to disk."""
        for name, model in self.models.items():
            if name != 'lstm':
                joblib.dump(model, f"{path}/{name}_model.joblib")
            else:
                model.save(f"{path}/{name}_model.h5")
        
        joblib.dump(self.scalers, f"{path}/scalers.joblib")
        joblib.dump(self.encoders, f"{path}/encoders.joblib")
        
        # Save feature list
        with open(f"{path}/features.txt", 'w') as f:
            f.write('\n'.join(self.features))
    
    def load_models(self, path: str):
        """Load all models and preprocessors from disk."""
        for name in ['logistic', 'rf', 'gb', 'voting']:
            self.models[name] = joblib.load(f"{path}/{name}_model.joblib")
        
        self.models['lstm'] = tf.keras.models.load_model(f"{path}/lstm_model.h5")
        self.scalers = joblib.load(f"{path}/scalers.joblib")
        self.encoders = joblib.load(f"{path}/encoders.joblib")
        
        # Load feature list
        with open(f"{path}/features.txt", 'r') as f:
            self.features = f.read().splitlines()

    def plot_feature_importance(self, save_dir: str = "figures"):
        """Plot and save feature importance from Random Forest model."""
        # Create the figures directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        importance = pd.DataFrame({
            'feature': self.features,
            'importance': self.models['rf'].feature_importances_
        })
        importance = importance.sort_values(by='importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importance)
        plt.title('Feature Importance from Random Forest')
        
        # Save the plot as a PNG file in the figures directory
        plot_path = os.path.join(save_dir, 'feature_importance.png')
        plt.savefig(plot_path)
        plt.close()  # Close the plot to free up memory
        self.logger.info(f"Feature importance plot saved to {plot_path}")
