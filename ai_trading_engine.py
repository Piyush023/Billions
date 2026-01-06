#!/usr/bin/env python3
"""
AI Trading Engine
Core machine learning models and prediction engine for intelligent trading decisions
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Import ML libraries with error handling for Python 3.13
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️ XGBoost not available - using alternative models")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("⚠️ LightGBM not available - using alternative models")
    LIGHTGBM_AVAILABLE = False

import joblib
import logging
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Tuple, Optional
import talib

class AITradingEngine:
    """Advanced AI engine for trading predictions and market analysis"""
    
    def __init__(self, config_path='config.json'):
        self.config = self.load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
        # Model configurations
        self.model_config = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 300,
                'max_depth': 8,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbosity': -1
            }
        }
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
    def load_config(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set for ML models"""
        try:
            features_df = df.copy()
            
            # Ensure data types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            
            # Price-based features
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['high_low_pct'] = (features_df['high'] - features_df['low']) / features_df['close']
            features_df['open_close_pct'] = (features_df['close'] - features_df['open']) / features_df['open']
            
            # Volume features
            features_df['volume_change'] = features_df['volume'].pct_change()
            features_df['price_volume'] = features_df['price_change'] * features_df['volume']
            
            # Technical indicators (converted to float64)
            close_prices = features_df['close'].values.astype(np.float64)
            high_prices = features_df['high'].values.astype(np.float64)
            low_prices = features_df['low'].values.astype(np.float64)
            volumes = features_df['volume'].values.astype(np.float64)
            
            # Trend indicators
            features_df['sma_5'] = talib.SMA(close_prices, timeperiod=5)
            features_df['sma_10'] = talib.SMA(close_prices, timeperiod=10)
            features_df['sma_20'] = talib.SMA(close_prices, timeperiod=20)
            features_df['sma_50'] = talib.SMA(close_prices, timeperiod=50)
            features_df['ema_12'] = talib.EMA(close_prices, timeperiod=12)
            features_df['ema_26'] = talib.EMA(close_prices, timeperiod=26)
            
            # Momentum indicators
            features_df['rsi'] = talib.RSI(close_prices, timeperiod=14)
            features_df['rsi_sma'] = talib.SMA(features_df['rsi'].values.astype(np.float64), timeperiod=5)
            features_df['macd'], features_df['macd_signal'], features_df['macd_hist'] = talib.MACD(close_prices)
            features_df['stoch_k'], features_df['stoch_d'] = talib.STOCH(high_prices, low_prices, close_prices)
            
            # Volatility indicators
            features_df['bb_upper'], features_df['bb_middle'], features_df['bb_lower'] = talib.BBANDS(close_prices)
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
            features_df['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)
            
            # Pattern recognition
            features_df['pattern_doji'] = talib.CDLDOJI(features_df['open'].values.astype(np.float64),
                                                       high_prices, low_prices, close_prices)
            features_df['pattern_hammer'] = talib.CDLHAMMER(features_df['open'].values.astype(np.float64),
                                                            high_prices, low_prices, close_prices)
            features_df['pattern_engulfing'] = talib.CDLENGULFING(features_df['open'].values.astype(np.float64),
                                                                  high_prices, low_prices, close_prices)
            
            # Relative position features
            features_df['close_sma20_ratio'] = features_df['close'] / features_df['sma_20']
            features_df['close_sma50_ratio'] = features_df['close'] / features_df['sma_50']
            features_df['bb_position'] = (features_df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # Lag features
            for lag in [1, 2, 3, 5]:
                features_df[f'price_change_lag_{lag}'] = features_df['price_change'].shift(lag)
                features_df[f'volume_change_lag_{lag}'] = features_df['volume_change'].shift(lag)
                features_df[f'rsi_lag_{lag}'] = features_df['rsi'].shift(lag)
            
            # Rolling statistics
            for window in [5, 10, 20]:
                features_df[f'price_volatility_{window}'] = features_df['price_change'].rolling(window).std()
                features_df[f'volume_avg_{window}'] = features_df['volume'].rolling(window).mean()
                features_df[f'price_max_{window}'] = features_df['close'].rolling(window).max()
                features_df[f'price_min_{window}'] = features_df['close'].rolling(window).min()
            
            # Time-based features
            if 'date' in features_df.columns or isinstance(features_df.index, pd.DatetimeIndex):
                if 'date' in features_df.columns:
                    features_df['hour'] = pd.to_datetime(features_df['date']).dt.hour
                    features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
                else:
                    features_df['hour'] = features_df.index.hour
                    features_df['day_of_week'] = features_df.index.dayofweek
            
            # Fill NaN values
            features_df = features_df.fillna(method='bfill').fillna(0)
            
            return features_df
            
        except Exception as e:
            self.logger.error(f"Feature creation failed: {e}")
            return df
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 1, 
                     profit_threshold: float = 0.02) -> pd.Series:
        """Create prediction labels based on future price movements"""
        try:
            future_prices = df['close'].shift(-prediction_horizon)
            current_prices = df['close']
            
            price_change = (future_prices - current_prices) / current_prices
            
            # Create labels: 1 for buy signal, 0 for hold/sell
            labels = (price_change > profit_threshold).astype(int)
            
            return labels
            
        except Exception as e:
            self.logger.error(f"Label creation failed: {e}")
            return pd.Series([0] * len(df))
    
    def prepare_training_data(self, symbol: str, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and labels for model training"""
        try:
            # Create features
            features_df = self.create_features(df)
            
            # Create labels
            labels = self.create_labels(features_df)
            
            # Select relevant features for training
            feature_columns = [col for col in features_df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'date']]
            
            X = features_df[feature_columns].copy()
            y = labels.copy()
            
            # Remove rows with NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]
            
            self.logger.info(f"Prepared training data for {symbol}: {len(X)} samples, {len(feature_columns)} features")
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed for {symbol}: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_ensemble_model(self, symbol: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble of ML models for the given symbol"""
        try:
            if len(X) < 100:
                self.logger.warning(f"Insufficient data for {symbol}: {len(X)} samples")
                return {}
            
            # Split data for training (keep last 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            models = {}
            scores = {}
            
            # Train Random Forest
            rf_model = RandomForestClassifier(**self.model_config['random_forest'])
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_val_scaled)
            rf_score = accuracy_score(y_val, rf_pred)
            
            models['random_forest'] = rf_model
            scores['random_forest'] = rf_score
            
            # Train XGBoost
            xgb_model = xgb.XGBClassifier(**self.model_config['xgboost'])
            xgb_model.fit(X_train_scaled, y_train)
            xgb_pred = xgb_model.predict(X_val_scaled)
            xgb_score = accuracy_score(y_val, xgb_pred)
            
            models['xgboost'] = xgb_model
            scores['xgboost'] = xgb_score
            
            # Train LightGBM
            lgb_model = lgb.LGBMClassifier(**self.model_config['lightgbm'])
            lgb_model.fit(X_train_scaled, y_train)
            lgb_pred = lgb_model.predict(X_val_scaled)
            lgb_score = accuracy_score(y_val, lgb_pred)
            
            models['lightgbm'] = lgb_model
            scores['lightgbm'] = lgb_score
            
            # Store models and scaler
            self.models[symbol] = models
            self.scalers[symbol] = scaler
            
            # Calculate feature importance
            feature_importance = {}
            feature_importance['random_forest'] = dict(zip(X.columns, rf_model.feature_importances_))
            feature_importance['xgboost'] = dict(zip(X.columns, xgb_model.feature_importances_))
            feature_importance['lightgbm'] = dict(zip(X.columns, lgb_model.feature_importances_))
            
            self.feature_importance[symbol] = feature_importance
            
            # Save models
            self.save_models(symbol)
            
            self.logger.info(f"Model training completed for {symbol}")
            self.logger.info(f"Model scores - RF: {rf_score:.3f}, XGB: {xgb_score:.3f}, LGB: {lgb_score:.3f}")
            
            return {
                'models': models,
                'scores': scores,
                'feature_importance': feature_importance
            }
            
        except Exception as e:
            self.logger.error(f"Model training failed for {symbol}: {e}")
            return {}
    
    def predict(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Generate AI predictions for the given symbol"""
        try:
            if symbol not in self.models:
                self.load_models(symbol)
                if symbol not in self.models:
                    return {'prediction': 0, 'confidence': 0, 'model_scores': {}}
            
            # Create features
            features_df = self.create_features(df)
            
            # Select feature columns (same as training)
            feature_columns = [col for col in features_df.columns 
                             if col not in ['open', 'high', 'low', 'close', 'volume', 'date']]
            
            X = features_df[feature_columns].iloc[-1:].copy()
            
            # Scale features
            X_scaled = self.scalers[symbol].transform(X)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, model in self.models[symbol].items():
                pred = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0]
                
                predictions[model_name] = pred
                probabilities[model_name] = pred_proba[1] if len(pred_proba) > 1 else pred_proba[0]
            
            # Ensemble prediction (voting)
            ensemble_prediction = int(sum(predictions.values()) > len(predictions) / 2)
            
            # Calculate confidence as average probability
            ensemble_confidence = np.mean(list(probabilities.values())) * 100
            
            # Adjust confidence based on model agreement
            agreement = len([p for p in predictions.values() if p == ensemble_prediction]) / len(predictions)
            ensemble_confidence *= agreement
            
            return {
                'prediction': ensemble_prediction,
                'confidence': ensemble_confidence,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'agreement': agreement
            }
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol}: {e}")
            return {'prediction': 0, 'confidence': 0, 'model_scores': {}}
    
    def get_feature_importance(self, symbol: str, top_n: int = 10) -> Dict:
        """Get top feature importance for the symbol"""
        try:
            if symbol not in self.feature_importance:
                return {}
            
            # Average feature importance across models
            all_features = set()
            for model_importance in self.feature_importance[symbol].values():
                all_features.update(model_importance.keys())
            
            avg_importance = {}
            for feature in all_features:
                importance_values = []
                for model_importance in self.feature_importance[symbol].values():
                    if feature in model_importance:
                        importance_values.append(model_importance[feature])
                
                if importance_values:
                    avg_importance[feature] = np.mean(importance_values)
            
            # Sort and get top N
            sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features[:top_n])
            
        except Exception as e:
            self.logger.error(f"Feature importance extraction failed for {symbol}: {e}")
            return {}
    
    def save_models(self, symbol: str):
        """Save trained models to disk"""
        try:
            symbol_dir = f"models/{symbol}"
            os.makedirs(symbol_dir, exist_ok=True)
            
            # Save models
            for model_name, model in self.models[symbol].items():
                joblib.dump(model, f"{symbol_dir}/{model_name}.joblib")
            
            # Save scaler
            joblib.dump(self.scalers[symbol], f"{symbol_dir}/scaler.joblib")
            
            # Save feature importance
            with open(f"{symbol_dir}/feature_importance.json", 'w') as f:
                json.dump(self.feature_importance[symbol], f, indent=2)
            
            self.logger.info(f"Models saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Model saving failed for {symbol}: {e}")
    
    def load_models(self, symbol: str):
        """Load trained models from disk"""
        try:
            symbol_dir = f"models/{symbol}"
            
            if not os.path.exists(symbol_dir):
                self.logger.warning(f"No saved models found for {symbol}")
                return
            
            # Load models
            models = {}
            for model_name in ['random_forest', 'xgboost', 'lightgbm']:
                model_path = f"{symbol_dir}/{model_name}.joblib"
                if os.path.exists(model_path):
                    models[model_name] = joblib.load(model_path)
            
            # Load scaler
            scaler_path = f"{symbol_dir}/scaler.joblib"
            if os.path.exists(scaler_path):
                self.scalers[symbol] = joblib.load(scaler_path)
            
            # Load feature importance
            importance_path = f"{symbol_dir}/feature_importance.json"
            if os.path.exists(importance_path):
                with open(importance_path, 'r') as f:
                    self.feature_importance[symbol] = json.load(f)
            
            self.models[symbol] = models
            self.logger.info(f"Models loaded for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Model loading failed for {symbol}: {e}")
    
    def retrain_models(self, symbol: str, df: pd.DataFrame) -> bool:
        """Retrain models with new data"""
        try:
            X, y = self.prepare_training_data(symbol, df)
            
            if len(X) < 100:
                self.logger.warning(f"Insufficient data for retraining {symbol}")
                return False
            
            result = self.train_ensemble_model(symbol, X, y)
            
            return len(result) > 0
            
        except Exception as e:
            self.logger.error(f"Model retraining failed for {symbol}: {e}")
            return False
    
    def get_model_performance(self, symbol: str) -> Dict:
        """Get performance metrics for trained models"""
        try:
            if symbol not in self.models:
                return {}
            
            performance = {
                'symbol': symbol,
                'models_trained': list(self.models[symbol].keys()),
                'feature_count': len(self.feature_importance.get(symbol, {}).get('random_forest', {})),
                'top_features': list(self.get_feature_importance(symbol, 5).keys())
            }
            
            return performance
            
        except Exception as e:
            self.logger.error(f"Performance metrics extraction failed for {symbol}: {e}")
            return {}