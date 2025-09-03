import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class WideDeepDataset(Dataset):
    def __init__(self, data, deep_categorical_features, continuous_features, wide_features, target_col='target'):
        self.data = data.reset_index(drop=True)
        self.deep_categorical_features = deep_categorical_features
        self.continuous_features = continuous_features
        self.wide_features = wide_features
        self.target_col = target_col

        self.wide_encoders = {}
        self.wide_feature_dims = {}

        # ✅ Pre-encode wide features ONCE here
        for feature in wide_features:
            if feature in self.data.columns:
                encoder = LabelEncoder()
                feature_data = self.data[feature].fillna('unknown').astype(str)
                self.data[feature] = encoder.fit_transform(feature_data)   # overwrite with ints
                self.wide_encoders[feature] = encoder
                self.wide_feature_dims[feature] = len(encoder.classes_)
                print(f"Wide feature {feature}: {len(encoder.classes_)} unique values")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        deep_categorical = torch.tensor([
            int(row[feature]) if pd.notna(row[feature]) else 0
            for feature in self.deep_categorical_features
        ], dtype=torch.long)

        continuous = torch.tensor([
            float(row[feature]) if pd.notna(row[feature]) else 0.0
            for feature in self.continuous_features
        ], dtype=torch.float32)

        # ✅ Now wide features are already ints
        wide_feature_indices = {feature: int(row[feature]) for feature in self.wide_features}

        target = torch.tensor(float(row[self.target_col]), dtype=torch.float32)

        return {
            'deep_categorical': deep_categorical,
            'continuous': continuous,
            'wide_feature_indices': wide_feature_indices,
            'target': target
        }



class WideComponent(nn.Module):
    def __init__(self, wide_feature_dims):
        super(WideComponent, self).__init__()
        self.wide_embeddings = nn.ModuleDict()
        total_wide_dim = 0
        for feature_name, vocab_size in wide_feature_dims.items():
            embed_dim = min(16, max(4, int(np.sqrt(vocab_size))))
            self.wide_embeddings[feature_name] = nn.Embedding(vocab_size, embed_dim)
            total_wide_dim += embed_dim
            nn.init.normal_(self.wide_embeddings[feature_name].weight, mean=0, std=0.01)
        self.wide_linear = nn.Linear(total_wide_dim, 1, bias=False)
        nn.init.normal_(self.wide_linear.weight, mean=0, std=0.01)

    def forward(self, wide_feature_indices):
        wide_embeddings = []
        for feature_name, embedding_layer in self.wide_embeddings.items():
            if feature_name in wide_feature_indices:
                feature_indices = torch.clamp(wide_feature_indices[feature_name], 0, embedding_layer.num_embeddings - 1)
                embed_output = embedding_layer(feature_indices)
                wide_embeddings.append(embed_output)
        if wide_embeddings:
            wide_concat = torch.cat(wide_embeddings, dim=1)
        else:
            batch_size = list(wide_feature_indices.values())[0].size(0)
            wide_concat = torch.zeros(batch_size, 1, device=next(self.parameters()).device)
        return self.wide_linear(wide_concat)


class DeepComponent(nn.Module):
    def __init__(self, categorical_vocab_sizes, embedding_dims, continuous_dim, hidden_dims=[1024, 512, 256]):
        super(DeepComponent, self).__init__()
        self.embeddings = nn.ModuleDict()
        total_embedding_dim = 0
        for feature, vocab_size in categorical_vocab_sizes.items():
            embed_dim = embedding_dims.get(feature, min(50, (vocab_size + 1) // 2))
            self.embeddings[feature] = nn.Embedding(vocab_size + 1, embed_dim)
            total_embedding_dim += embed_dim
            nn.init.normal_(self.embeddings[feature].weight, mean=0, std=0.01)
        input_dim = total_embedding_dim + continuous_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.deep_network = nn.Sequential(*layers)
        for layer in self.deep_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, deep_categorical, continuous):
        embedding_outputs = []
        for i, (feature, embedding_layer) in enumerate(self.embeddings.items()):
            if i < deep_categorical.size(1):
                feature_input = torch.clamp(deep_categorical[:, i], 0, embedding_layer.num_embeddings - 1)
                embedding_outputs.append(embedding_layer(feature_input))
        embeddings_concat = torch.cat(embedding_outputs, dim=1) if embedding_outputs else torch.zeros(deep_categorical.size(0), 0)
        deep_input = torch.cat([embeddings_concat, continuous], dim=1) if continuous.size(1) > 0 else embeddings_concat
        return self.deep_network(deep_input)


class WideDeepModel(nn.Module):
    def __init__(self, categorical_vocab_sizes, embedding_dims, continuous_dim, wide_feature_dims, hidden_dims=[512, 256, 128]):
        super(WideDeepModel, self).__init__()
        self.wide = WideComponent(wide_feature_dims)
        self.deep = DeepComponent(categorical_vocab_sizes, embedding_dims, continuous_dim, hidden_dims)
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, deep_categorical, continuous, wide_feature_indices):
        wide_output = self.wide(wide_feature_indices)
        deep_output = self.deep(deep_categorical, continuous)
        combined_output = wide_output + deep_output + self.output_bias
        return torch.sigmoid(combined_output)


class WideDeepTrainer:
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.wide_optimizer = optim.Adagrad(self.model.wide.parameters(), lr=0.01)
        self.deep_optimizer = optim.Adam(list(self.model.deep.parameters()) + [self.model.output_bias], lr=0.001)
        self.criterion = nn.BCELoss()
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        for batch in tqdm(self.train_loader, desc="Training", total=len(self.train_loader)):
            deep_categorical = batch['deep_categorical'].to(self.device)
            continuous = batch['continuous'].to(self.device)
            targets = batch['target'].to(self.device).unsqueeze(1)
            wide_feature_indices = {k: v.to(self.device) for k, v in batch['wide_feature_indices'].items()}
            self.wide_optimizer.zero_grad()
            self.deep_optimizer.zero_grad()
            outputs = self.model(deep_categorical, continuous, wide_feature_indices)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.wide_optimizer.step()
            self.deep_optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_predictions, all_targets = [], []
        num_batches = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", total=len(self.val_loader)):
                deep_categorical = batch['deep_categorical'].to(self.device)
                continuous = batch['continuous'].to(self.device)
                targets = batch['target'].to(self.device).unsqueeze(1)
                wide_feature_indices = {k: v.to(self.device) for k, v in batch['wide_feature_indices'].items()}
                outputs = self.model(deep_categorical, continuous, wide_feature_indices)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                num_batches += 1
        return total_loss / num_batches, roc_auc_score(all_targets, all_predictions), all_predictions, all_targets
    def train(self, num_epochs):
        print(f"Training Wide & Deep model for {num_epochs} epochs...")
        best_auc = 0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, val_auc, _, _ = self.validate()
            self.val_losses.append(val_loss)
            self.val_aucs.append(val_auc)

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val AUC: {val_auc:.4f}")

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_wide_deep_model.pt')
                print(f"New best model saved! AUC: {best_auc:.4f}")

        print(f"\nTraining completed! Best AUC: {best_auc:.4f}")
        return best_auc
    def evaluate_model(self, test_loader):
        print("\nEvaluating model on test set...")
        self.model.load_state_dict(torch.load('best_wide_deep_model.pt'))
        self.model.eval()
        all_predictions, all_targets = [], []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                deep_categorical = batch['deep_categorical'].to(self.device)
                continuous = batch['continuous'].to(self.device)
                targets = batch['target'].to(self.device)
                wide_feature_indices = {k: v.to(self.device) for k, v in batch['wide_feature_indices'].items()}
                outputs = self.model(deep_categorical, continuous, wide_feature_indices)
                all_predictions.extend(outputs.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
        predictions_binary = (np.array(all_predictions) > 0.5).astype(int)
        auc = roc_auc_score(all_targets, all_predictions)
        accuracy = accuracy_score(all_targets, predictions_binary)
        precision = precision_score(all_targets, predictions_binary)
        recall = recall_score(all_targets, predictions_binary)
        print(f"\nTest Results:\nAUC: {auc:.4f}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}")
        return {'auc': auc, 'accuracy': accuracy, 'precision': precision, 'recall': recall, 'predictions': all_predictions, 'targets': all_targets}


def custom_collate_fn(batch):
    print("Batch size:", len(batch))   # ✅ debug
    deep_categorical = torch.stack([b['deep_categorical'] for b in batch])
    continuous = torch.stack([b['continuous'] for b in batch])
    targets = torch.stack([b['target'] for b in batch])
    feature_names = batch[0]['wide_feature_indices'].keys()
    wide_feature_indices = {
        f: torch.tensor([b['wide_feature_indices'][f] for b in batch], dtype=torch.long)
        for f in feature_names
    }
    print("Collated batch!")   # ✅ debug
    return {
        'deep_categorical': deep_categorical,
        'continuous': continuous,
        'wide_feature_indices': wide_feature_indices,
        'target': targets
    }


def main():
    print("Wide & Deep Learning Model Training")
    print("="*50)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_dir = "/content/processed_data"  # ✅ adjust if mounted from Drive

    try:
        df = pd.read_parquet(os.path.join(data_dir, "processed_reviews.parquet"))
        with open(os.path.join(data_dir, "vocabularies.pkl"), "rb") as f:
            vocabularies = pickle.load(f)
    except FileNotFoundError:
        print("Error: Processed data not found. Please upload or run amazon_dataset_preprocess.py first.")
        raise

    deep_categorical_features = [f for f in ['user_id', 'item_id', 'review_year', 'review_month'] if f in df.columns]
    continuous_features = [f for f in ['user_avg_rating', 'user_review_count', 'item_avg_rating', 'item_review_count', 'review_length', 'review_word_count'] if f in df.columns]
    wide_features = [f for f in ['user_item_cross', 'user_year_cross', 'item_year_cross', 'user_item_rating_cross'] if f in df.columns]

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    train_dataset = WideDeepDataset(train_df, deep_categorical_features, continuous_features, wide_features)
    test_dataset = WideDeepDataset(test_df, deep_categorical_features, continuous_features, wide_features)

    wide_feature_dims = train_dataset.wide_feature_dims

    pin = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=custom_collate_fn, num_workers=0, pin_memory=False)
    

    print("Dataset length:", len(train_dataset))
    print(train_dataset[0])  # should return instantly

    
    categorical_vocab_sizes = {
        'user_id': len(vocabularies['user_vocab']),
        'item_id': len(vocabularies['item_vocab']),
        'review_year': df['review_year'].max() - df['review_year'].min() + 1 if 'review_year' in df.columns else 10,
        'review_month': 12
    }
    embedding_dims = {'user_id': 32, 'item_id': 32, 'review_year': 8, 'review_month': 4}

    model = WideDeepModel(categorical_vocab_sizes, embedding_dims, len(continuous_features), wide_feature_dims, hidden_dims=[512, 256, 128])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = WideDeepTrainer(model, train_loader, test_loader, device)
    best_auc = trainer.train(num_epochs=10)
    trainer.evaluate_model(test_loader)

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)


if __name__ == "__main__":
    main()
