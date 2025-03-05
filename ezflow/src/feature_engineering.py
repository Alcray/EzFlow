"""
Feature engineering module for the ezflow framework.

This module contains functionality for creating, transforming, encoding, and scaling features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature engineering class for handling data transformations, encodings, and scaling.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the FeatureEngineer with configuration.
        
        Args:
            config (Dict): Configuration dictionary with feature engineering parameters.
        """
        self.config = config
        
        # Initialize transformers
        self.numeric_transformer = None
        self.categorical_transformer = None
        self.column_transformer = None
        
        # Track fitted attributes
        self.fitted = False
        
        logger.info("FeatureEngineer initialized")
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: DataFrame with new features.
        """
        logger.info("Creating new features...")
        
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Example feature creation: date features
        date_columns = df_features.select_dtypes(include=['datetime64']).columns.tolist()
        
        for date_col in date_columns:
            logger.info(f"Extracting date features from {date_col}")
            
            # Extract date components
            df_features[f"{date_col}_year"] = df_features[date_col].dt.year
            df_features[f"{date_col}_month"] = df_features[date_col].dt.month
            df_features[f"{date_col}_day"] = df_features[date_col].dt.day
            df_features[f"{date_col}_dayofweek"] = df_features[date_col].dt.dayofweek
            df_features[f"{date_col}_quarter"] = df_features[date_col].dt.quarter
            
            # Add seasonality indicators
            df_features[f"{date_col}_is_weekend"] = df_features[f"{date_col}_dayofweek"].isin([5, 6]).astype(int)
            df_features[f"{date_col}_is_month_start"] = df_features[date_col].dt.is_month_start.astype(int)
            df_features[f"{date_col}_is_month_end"] = df_features[date_col].dt.is_month_end.astype(int)
        
        # Example: Interaction features
        # This can be customized based on domain knowledge
        numeric_cols = df_features.select_dtypes(include=np.number).columns.tolist()
        
        # Add some interactions between numeric features if requested
        if self.config.get('CREATE_INTERACTIONS', False) and len(numeric_cols) >= 2:
            logger.info("Creating interaction features between numeric columns")
            
            # Limit the number of interactions to avoid too many features
            max_interactions = min(10, (len(numeric_cols) * (len(numeric_cols) - 1)) // 2)
            interaction_count = 0
            
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    if interaction_count >= max_interactions:
                        break
                    
                    # Add multiplication interaction
                    df_features[f"{col1}_x_{col2}"] = df_features[col1] * df_features[col2]
                    interaction_count += 1
                    
                    # Add ratio interaction if requested and denominator is not zero
                    if self.config.get('CREATE_RATIOS', False):
                        # Avoid division by zero
                        mask = df_features[col2] != 0
                        col_name = f"{col1}_div_{col2}"
                        df_features[col_name] = np.nan
                        df_features.loc[mask, col_name] = df_features.loc[mask, col1] / df_features.loc[mask, col2]
                        interaction_count += 1
        
        logger.info(f"Feature creation completed. New shape: {df_features.shape}")
        return df_features
    
    def setup_preprocessing(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str],
        numeric_strategy: str = 'mean',
        categorical_strategy: str = 'most_frequent',
        scaling: str = 'standard'
    ) -> None:
        """
        Set up preprocessing pipelines for numeric and categorical features.
        
        Args:
            numeric_features (List[str]): List of numeric feature names.
            categorical_features (List[str]): List of categorical feature names.
            numeric_strategy (str): Strategy for imputing missing numeric values.
            categorical_strategy (str): Strategy for imputing missing categorical values.
            scaling (str): Scaling method ('standard', 'minmax', or None).
        """
        logger.info("Setting up preprocessing pipelines")
        
        # Numeric preprocessing pipeline
        numeric_steps = [
            ('imputer', SimpleImputer(strategy=numeric_strategy))
        ]
        
        if scaling == 'standard':
            numeric_steps.append(('scaler', StandardScaler()))
        elif scaling == 'minmax':
            numeric_steps.append(('scaler', MinMaxScaler()))
        
        self.numeric_transformer = Pipeline(steps=numeric_steps)
        
        # Categorical preprocessing pipeline
        self.categorical_transformer = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy)),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ]
        )
        
        # Column transformer to apply the appropriate preprocessing to each column
        transformers = []
        
        if numeric_features:
            transformers.append(('num', self.numeric_transformer, numeric_features))
        
        if categorical_features:
            transformers.append(('cat', self.categorical_transformer, categorical_features))
        
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
        
        logger.info(f"Preprocessing setup completed with {len(numeric_features)} numeric and {len(categorical_features)} categorical features")
    
    def detect_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Automatically detect numeric and categorical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            Tuple[List[str], List[str]]: Lists of numeric and categorical column names.
        """
        logger.info("Detecting feature types...")
        
        # Numeric features: all numeric columns except those with few unique values
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Exclude columns that are likely categorical encoded as numbers
        categorical_threshold = min(10, max(3, len(df) // 100))
        
        categorical_cols = []
        true_numeric_cols = []
        
        for col in numeric_cols:
            n_unique = df[col].nunique()
            if n_unique < categorical_threshold:
                categorical_cols.append(col)
            else:
                true_numeric_cols.append(col)
        
        # Categorical features: object, category, and boolean columns
        categorical_cols.extend(df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist())
        
        logger.info(f"Detected {len(true_numeric_cols)} numeric and {len(categorical_cols)} categorical features")
        return true_numeric_cols, categorical_cols
    
    def fit_transform(self, df: pd.DataFrame, numeric_features: Optional[List[str]] = None, 
                    categorical_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fit and transform the data using the preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            numeric_features (Optional[List[str]]): List of numeric feature names.
            categorical_features (Optional[List[str]]): List of categorical feature names.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        logger.info("Fitting and transforming data...")
        
        # Auto-detect feature types if not provided
        if numeric_features is None or categorical_features is None:
            auto_numeric, auto_categorical = self.detect_feature_types(df)
            numeric_features = numeric_features or auto_numeric
            categorical_features = categorical_features or auto_categorical
        
        # Set up preprocessing if not already done
        if self.column_transformer is None:
            self.setup_preprocessing(
                numeric_features, 
                categorical_features,
                numeric_strategy=self.config.get('NUMERIC_IMPUTE_STRATEGY', 'mean'),
                categorical_strategy=self.config.get('CATEGORICAL_IMPUTE_STRATEGY', 'most_frequent'),
                scaling=self.config.get('SCALING_METHOD', 'standard')
            )
        
        # Fit and transform
        transformed_array = self.column_transformer.fit_transform(df)
        self.fitted = True
        
        # Get transformed feature names
        feature_names = self._get_feature_names_after_transform(df.columns)
        
        # Convert back to DataFrame
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names)
        
        logger.info(f"Data transformed. New shape: {transformed_df.shape}")
        return transformed_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            
        Returns:
            pd.DataFrame: Transformed DataFrame.
            
        Raises:
            ValueError: If called before fitting.
        """
        if not self.fitted or self.column_transformer is None:
            raise ValueError("Pipeline has not been fitted. Call fit_transform first.")
        
        logger.info("Transforming data using fitted pipeline...")
        
        # Apply transform
        transformed_array = self.column_transformer.transform(df)
        
        # Get transformed feature names
        feature_names = self._get_feature_names_after_transform(df.columns)
        
        # Convert back to DataFrame
        transformed_df = pd.DataFrame(transformed_array, columns=feature_names)
        
        logger.info(f"Data transformed. New shape: {transformed_df.shape}")
        return transformed_df
    
    def _get_feature_names_after_transform(self, original_columns):
        """Get feature names after transformation."""
        # For sklearn < 1.0, column_transformer doesn't have get_feature_names_out
        if hasattr(self.column_transformer, 'get_feature_names_out'):
            return self.column_transformer.get_feature_names_out()
        
        # Manual approach for older sklearn versions
        # This is a simplified version and may need adjustments
        transformed_features = []
        
        for name, transformer, features in self.column_transformer.transformers_:
            if name == 'remainder':
                # Passthrough columns
                transformed_features.extend([col for col in original_columns if col not in
                                           self.column_transformer._columntransformer__column_names])
            elif name == 'num':
                # Numeric columns stay the same
                transformed_features.extend(features)
            elif name == 'cat':
                # OneHotEncoder creates multiple columns
                for feature in features:
                    transformed_features.append(f"{feature}_encoded")
        
        return transformed_features
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str], method: str = 'onehot') -> pd.DataFrame:
        """
        Encode categorical variables with various methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to encode.
            method (str): Encoding method ('onehot', 'label', 'target').
            
        Returns:
            pd.DataFrame: DataFrame with encoded features.
        """
        logger.info(f"Encoding {len(columns)} categorical columns using {method} encoding")
        
        df_encoded = df.copy()
        
        if method == 'onehot':
            # One-hot encoding
            encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            encoded_array = encoder.fit_transform(df_encoded[columns])
            
            # Create DataFrame with encoded columns
            encoded_cols = [f"{col}_{cat}" for i, col in enumerate(columns) 
                          for cat in encoder.categories_[i]]
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_encoded.index)
            
            # Drop original columns and concatenate encoded ones
            df_encoded = pd.concat([df_encoded.drop(columns=columns), encoded_df], axis=1)
            
        elif method == 'label':
            # Label encoding
            for col in columns:
                encoder = LabelEncoder()
                df_encoded[f"{col}_encoded"] = encoder.fit_transform(df_encoded[col].astype(str))
                df_encoded = df_encoded.drop(columns=[col])
        
        elif method == 'target':
            # Target encoding would require target variable
            # For simplicity, fallback to label encoding here
            logger.warning("Target encoding not implemented yet. Using label encoding instead.")
            for col in columns:
                encoder = LabelEncoder()
                df_encoded[f"{col}_encoded"] = encoder.fit_transform(df_encoded[col].astype(str))
                df_encoded = df_encoded.drop(columns=[col])
        
        else:
            raise ValueError(f"Unknown encoding method: {method}")
        
        logger.info(f"Encoding completed. New shape: {df_encoded.shape}")
        return df_encoded
    
    def scale_features(self, df: pd.DataFrame, columns: List[str], method: str = 'standard') -> pd.DataFrame:
        """
        Scale numeric features using various methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            columns (List[str]): Columns to scale.
            method (str): Scaling method ('standard', 'minmax').
            
        Returns:
            pd.DataFrame: DataFrame with scaled features.
        """
        logger.info(f"Scaling {len(columns)} numeric columns using {method} scaling")
        
        df_scaled = df.copy()
        
        if method == 'standard':
            # StandardScaler: (X - mean) / std, output has mean=0 and std=1
            scaler = StandardScaler()
        elif method == 'minmax':
            # MinMaxScaler: (X - min) / (max - min), output is in range [0, 1]
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Apply scaling
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        
        logger.info("Scaling completed")
        return df_scaled 