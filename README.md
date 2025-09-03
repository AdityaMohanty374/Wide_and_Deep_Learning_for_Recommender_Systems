# **Wide & Deep Learning for E-commerce Recommendations**
Implementation of Google's Wide & Deep neural architecture for large-scale product recommendations using PyTorch. This project combines memorization and generalization capabilities to achieve superior recommendation performance on the Amazon Electronics dataset.
## Results Summary
Show Image
## Key Performance Metrics

AUC Score: 0.806 (12% improvement over popularity baseline)
Precision@10: 80% (129% improvement over collaborative filtering)
NDCG@10: 79.5% (32% improvement over simple ranking)
Catalog Coverage: 99.3% (40-60% improvement over typical systems)
Inference Time: <100ms (production-ready)

## Dataset Scale

Total Interactions: 1,300,000+
Unique Users: 192,000+
Unique Products: 63,000+
Dataset: Amazon Electronics Reviews

## Architecture Overview
### Wide Component

Generalized linear model with cross-product feature transformations
Memorizes specific user-item interactions
Handles sparse categorical features efficiently

### Deep Component

Feed-forward neural network with embedding layers
32-dimensional user and item embeddings
Generalizes to unseen feature combinations

### Joint Training

Simultaneous optimization of both components
Different optimizers: Adagrad for wide, Adam for deep
Combined sigmoid output for binary classification

## Project Structure
```
wide-deep-recommender/
├── amazon_preprocessing.py      # Data preprocessing pipeline
├── wide_deep_model.py          # Model architecture and training
├── model_evaluation.py         # Comprehensive evaluation
├── processed_data/             # Processed datasets
│   ├── processed_reviews.parquet
│   └── vocabularies.pkl
├── best_wide_deep_model.pt     # Trained model weights
├── results.png                 # Evaluation visualizations
└── README.md
```
## Features Engineered
### Deep Component Features

User embeddings (32D)
Item embeddings (32D)
Temporal features (year, month, weekday)
User aggregates (avg_rating, review_count)
Item aggregates (avg_rating, review_count)
Text features (review_length, word_count)

### Wide Component Features

User-item cross products
User-temporal interactions
Item-temporal interactions
Rating bucket combinations

## Quick Start
1. Data Preprocessing
`bashpython amazon_preprocessing.py`
2. Model Training
`bashpython wide_deep_model.py`
3. Model Evaluation
`bashpython model_evaluation.py`
Requirements
`bashpip install torch pandas numpy scikit-learn matplotlib seaborn pyarrow tqdm`

## Model Performance
MetricValueImprovementAUC0.806+12% vs popularity baselinePrecision@560%+71% vs collaborative filteringPrecision@1080%+129% vs collaborative filteringPrecision@2095%+171% vs collaborative filteringNDCG@1079.5%+32% vs simple rankingCoverage99.3%+40-60% vs typical systems
## Business Impact
Revenue Potential

Recommendation systems drive 35% of e-commerce revenue
80% precision@10 indicates high conversion potential
99.3% catalog coverage promotes long-tail products

## User Experience

Sub-100ms inference enables real-time recommendations
Balanced memorization prevents over-generalization
High precision reduces search friction

## Technical Scalability

Handles 1.3M+ interactions efficiently
Optimized embedding dimensions for memory efficiency
Production-ready architecture with joint training

Implementation Highlights
Data Engineering

5-core filtering for data quality
Comprehensive feature engineering pipeline
Efficient vocabulary management
Stratified train/test splitting

## Model Optimization

Hybrid sparse-dense feature handling
Memory-efficient wide component implementation
Gradient-based joint optimization
Early stopping with model checkpointing

## Evaluation Framework

Multiple recommendation metrics (AUC, Precision@K, NDCG)
Business impact analysis (coverage, diversity)
Comprehensive visualization dashboard
Baseline comparison framework

## Technical Stack

Framework: PyTorch
Data Processing: pandas, NumPy
Evaluation: scikit-learn
Visualization: matplotlib, seaborn
Storage: Parquet (efficient columnar storage)

## Key Learnings

Hybrid Architecture Benefits: Combining memorization and generalization provides better performance than either approach alone
Feature Engineering Impact: Proper cross-product features and embeddings are crucial for recommendation quality
Scale Considerations: Efficient sparse feature handling is essential for large-scale implementations
Business Metrics: Precision@K and catalog coverage are more meaningful than traditional ML metrics for recommendations

## Future Enhancements

Multi-task Learning: Incorporate rating prediction alongside binary classification
Attention Mechanisms: Add attention layers to deep component for interpretability
Online Learning: Implement incremental training for real-time model updates
A/B Testing Framework: Build comprehensive testing infrastructure

## References

[Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) - Google Research
[Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/) - UCSD Dataset
