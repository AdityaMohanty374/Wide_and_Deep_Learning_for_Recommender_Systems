import pandas as pd
import numpy as np
import json
import gzip
from collections import Counter
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class AmazonDataProcessor:
    def __init__(self, data_path='reviews_Electronics_5.json.gz'):
        self.data_path = data_path
        self.df = None
        self.user_vocab = {}
        self.item_vocab = {}
        self.category_vocab = {}
        self.brand_vocab = {}
        
    def download_and_extract_data(self):
        """
        Download Amazon Electronics dataset if not exists
        """
        if not os.path.exists(self.data_path):
            print("Dataset not found locally. Please download from:")
            print("http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz")
            print("Place it in the same directory as this script.")
            return False
        return True
    
    def load_data(self, sample_size=None):
        """
        Load and parse Amazon Electronics JSON data
        """
        print("Loading Amazon Electronics dataset...")
        
        data = []
        with gzip.open(self.data_path, 'rt', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if sample_size and i >= sample_size:
                    break
                try:
                    data.append(json.loads(line))
                except:
                    continue
                    
                if i % 100000 == 0:
                    print(f"Processed {i} records...")
        
        self.df = pd.DataFrame(data)
        print(f"Loaded {len(self.df)} reviews")
        return self.df
    
    def explore_data(self):
        """
        Basic data exploration
        """
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Basic statistics
        print(f"\nUnique users: {self.df['reviewerID'].nunique():,}")
        print(f"Unique products: {self.df['asin'].nunique():,}")
        print(f"Rating distribution:")
        print(self.df['overall'].value_counts().sort_index())
        
        # Missing values
        print(f"\nMissing values:")
        print(self.df.isnull().sum())
        
        # Date range
        if 'unixReviewTime' in self.df.columns:
            self.df['review_date'] = pd.to_datetime(self.df['unixReviewTime'], unit='s')
            print(f"\nDate range: {self.df['review_date'].min()} to {self.df['review_date'].max()}")
        
        return self.df.describe()
    
    def create_target_variable(self):
        """
        Create binary target: 1 if rating >= 4, 0 otherwise
        """
        self.df['target'] = (self.df['overall'] >= 4).astype(int)
        
        print(f"\nTarget distribution:")
        print(self.df['target'].value_counts())
        print(f"Positive rate: {self.df['target'].mean():.3f}")
        
        return self.df
    
    def filter_data(self, min_user_interactions=5, min_item_interactions=5):
        """
        Filter users and items with minimum interactions (5-core filtering)
        """
        print(f"\nFiltering data (min {min_user_interactions} interactions per user/item)...")
        
        initial_size = len(self.df)
        
        # Iterative filtering until convergence
        prev_users, prev_items = 0, 0
        while True:
            # Filter users
            user_counts = self.df['reviewerID'].value_counts()
            valid_users = user_counts[user_counts >= min_user_interactions].index
            self.df = self.df[self.df['reviewerID'].isin(valid_users)]
            
            # Filter items
            item_counts = self.df['asin'].value_counts()
            valid_items = item_counts[item_counts >= min_item_interactions].index
            self.df = self.df[self.df['asin'].isin(valid_items)]
            
            curr_users = self.df['reviewerID'].nunique()
            curr_items = self.df['asin'].nunique()
            
            if curr_users == prev_users and curr_items == prev_items:
                break
                
            prev_users, prev_items = curr_users, curr_items
        
        print(f"After filtering: {len(self.df):,} reviews ({len(self.df)/initial_size:.1%} of original)")
        print(f"Users: {self.df['reviewerID'].nunique():,}")
        print(f"Items: {self.df['asin'].nunique():,}")
        
        return self.df
    
    def create_vocabularies(self):
        """
        Create vocabularies for categorical features
        """
        print("\nCreating vocabularies...")
        
        # User vocabulary
        unique_users = self.df['reviewerID'].unique()
        self.user_vocab = {user: idx for idx, user in enumerate(unique_users)}
        
        # Item vocabulary  
        unique_items = self.df['asin'].unique()
        self.item_vocab = {item: idx for idx, item in enumerate(unique_items)}
        
        # Category vocabulary (extract from item categories if available)
        if 'categories' in self.df.columns:
            all_categories = []
            for cats in self.df['categories'].dropna():
                if isinstance(cats, list) and len(cats) > 0:
                    all_categories.extend(cats[0] if isinstance(cats[0], list) else [cats[0]])
            
            unique_categories = list(set(all_categories))
            self.category_vocab = {cat: idx for idx, cat in enumerate(unique_categories)}
        
        # Brand vocabulary (if available)
        if 'brand' in self.df.columns:
            unique_brands = self.df['brand'].dropna().unique()
            self.brand_vocab = {brand: idx for idx, brand in enumerate(unique_brands)}
        
        print(f"Vocabularies created:")
        print(f"- Users: {len(self.user_vocab):,}")
        print(f"- Items: {len(self.item_vocab):,}")
        print(f"- Categories: {len(self.category_vocab):,}")
        print(f"- Brands: {len(self.brand_vocab):,}")
        
        return self.user_vocab, self.item_vocab, self.category_vocab, self.brand_vocab
    
    def engineer_features(self):
        """
        Engineer features for Wide & Deep model
        """
        print("\nEngineering features...")
        
        # Map to vocabulary indices
        self.df['user_id'] = self.df['reviewerID'].map(self.user_vocab)
        self.df['item_id'] = self.df['asin'].map(self.item_vocab)
        
        # Time features
        if 'unixReviewTime' in self.df.columns:
            self.df['review_date'] = pd.to_datetime(self.df['unixReviewTime'], unit='s')
            self.df['review_year'] = self.df['review_date'].dt.year
            self.df['review_month'] = self.df['review_date'].dt.month
            self.df['review_weekday'] = self.df['review_date'].dt.weekday
        
        # Text features (basic)
        if 'reviewText' in self.df.columns:
            self.df['review_length'] = self.df['reviewText'].fillna('').str.len()
            self.df['review_word_count'] = self.df['reviewText'].fillna('').str.split().str.len()
        
        # User aggregated features
        user_stats = self.df.groupby('user_id').agg({
            'overall': ['mean', 'std', 'count'],
            'target': 'mean'
        }).round(3)
        
        user_stats.columns = ['user_avg_rating', 'user_rating_std', 'user_review_count', 'user_positive_rate']
        user_stats['user_rating_std'] = user_stats['user_rating_std'].fillna(0)
        
        # Item aggregated features  
        item_stats = self.df.groupby('item_id').agg({
            'overall': ['mean', 'std', 'count'],
            'target': 'mean'
        }).round(3)
        
        item_stats.columns = ['item_avg_rating', 'item_rating_std', 'item_review_count', 'item_positive_rate']
        item_stats['item_rating_std'] = item_stats['item_rating_std'].fillna(0)
        
        # Merge back
        self.df = self.df.merge(user_stats, left_on='user_id', right_index=True, how='left')
        self.df = self.df.merge(item_stats, left_on='item_id', right_index=True, how='left')
        
        print("Feature engineering completed!")
        return self.df
    
    def create_wide_features(self):
        """
        Create cross-product features for Wide component
        """
        print("\nCreating Wide component features...")
        
        wide_features = []
        
        # User-Item cross products
        self.df['user_item_cross'] = self.df['user_id'].astype(str) + '_' + self.df['item_id'].astype(str)
        
        # Time-based cross products
        if 'review_year' in self.df.columns:
            self.df['user_year_cross'] = self.df['user_id'].astype(str) + '_' + self.df['review_year'].astype(str)
            self.df['item_year_cross'] = self.df['item_id'].astype(str) + '_' + self.df['review_year'].astype(str)
        
        # Rating pattern cross products
        self.df['user_rating_bucket'] = pd.cut(self.df['user_avg_rating'], 
                                             bins=[0, 2, 3, 4, 5], 
                                             labels=['low', 'medium', 'high', 'very_high'])
        
        self.df['item_rating_bucket'] = pd.cut(self.df['item_avg_rating'], 
                                             bins=[0, 2, 3, 4, 5], 
                                             labels=['low', 'medium', 'high', 'very_high'])
        
        # Cross products of buckets
        self.df['user_item_rating_cross'] = (self.df['user_rating_bucket'].astype(str) + '_' + 
                                            self.df['item_rating_bucket'].astype(str))
        
        wide_feature_cols = ['user_item_cross', 'user_year_cross', 'item_year_cross', 'user_item_rating_cross']
        
        print(f"Created {len(wide_feature_cols)} wide feature columns")
        return wide_feature_cols
    
    def prepare_model_data(self):
        """
        Prepare final dataset for model training
        """
        print("\nPreparing model data...")
        
        # Select features for deep component (categorical for embeddings)
        deep_categorical_features = ['user_id', 'item_id']
        if 'review_year' in self.df.columns:
            deep_categorical_features.extend(['review_year', 'review_month', 'review_weekday'])
        
        # Select continuous features
        continuous_features = ['user_avg_rating', 'user_rating_std', 'user_review_count', 
                              'item_avg_rating', 'item_rating_std', 'item_review_count']
        
        if 'review_length' in self.df.columns:
            continuous_features.extend(['review_length', 'review_word_count'])
        
        # Normalize continuous features to [0, 1]
        for col in continuous_features:
            if col in self.df.columns:
                min_val, max_val = self.df[col].min(), self.df[col].max()
                if max_val > min_val:
                    self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
        
        # Create wide features
        wide_feature_cols = self.create_wide_features()
        
        # Final feature selection
        feature_columns = deep_categorical_features + continuous_features + wide_feature_cols + ['target']
        model_data = self.df[feature_columns].copy()
        
        print(f"Final dataset shape: {model_data.shape}")
        print(f"Deep categorical features: {deep_categorical_features}")
        print(f"Continuous features: {continuous_features}")
        print(f"Wide features: {wide_feature_cols}")
        
        return model_data, deep_categorical_features, continuous_features, wide_feature_cols
    
    def train_test_split_data(self, model_data, test_size=0.2, random_state=42):
        """
        Split data into train/test sets
        """
        print(f"\nSplitting data (test_size={test_size})...")
        
        X = model_data.drop('target', axis=1)
        y = model_data['target']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {len(X_train):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Train positive rate: {y_train.mean():.3f}")
        print(f"Test positive rate: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, save_dir='processed_data'):
        """
        Save all processed data and vocabularies
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\nSaving processed data to {save_dir}/...")
        
        # Save vocabularies
        with open(f'{save_dir}/vocabularies.pkl', 'wb') as f:
            pickle.dump({
                'user_vocab': self.user_vocab,
                'item_vocab': self.item_vocab,
                'category_vocab': self.category_vocab,
                'brand_vocab': self.brand_vocab
            }, f)
        
        # Save main dataframe
        self.df.to_parquet(f'{save_dir}/processed_reviews.parquet', index=False)
        
        print("Data saved successfully!")
        
        return True

def main():
    """
    Main preprocessing pipeline
    """
    print("Amazon Electronics Wide & Deep Data Preprocessing")
    print("="*60)
    
    # Initialize processor
    processor = AmazonDataProcessor('C:\\Users\\mohan\\OneDrive - vit.ac.in\\Documents\\cnn_mnist\\Wide And Deep Network\\reviews_Electronics_5.json.gz')
    
    # Check if data exists
    if not processor.download_and_extract_data():
        return
    
    # Load data (use sample_size=100000 for testing)
    processor.load_data(sample_size=None)  # Set to None for full dataset
    
    # Explore data
    processor.explore_data()
    
    # Create target variable
    processor.create_target_variable()
    
    # Filter data (5-core)
    processor.filter_data(min_user_interactions=5, min_item_interactions=5)
    
    # Create vocabularies
    processor.create_vocabularies()
    
    # Engineer features
    processor.engineer_features()
    
    # Prepare model data
    model_data, deep_categorical_features, continuous_features, wide_feature_cols = processor.prepare_model_data()
    
    # Train/test split
    X_train, X_test, y_train, y_test = processor.train_test_split_data(model_data)
    
    # Save processed data
    processor.save_processed_data()
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("Next steps:")
    print("1. Review the processed_data/ directory")
    print("2. Implement Wide & Deep model architecture") 
    print("3. Train and evaluate the model")
    
    return processor, model_data, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    processor, model_data, X_train, X_test, y_train, y_test = main()