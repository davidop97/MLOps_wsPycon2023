import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import wandb

def load(train_size=0.8, val_size=0.1, random_state=42):
    """
    Load Wine dataset from scikit-learn
    Dataset: 178 samples, 13 features, 3 classes (wine types)
    """
    
    # Cargar dataset de vinos
    data = load_wine()
    
    X, y = data.data, data.target
    
    # Convertir a tensores de PyTorch
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Primera división: train+val vs test
    test_size = 1 - train_size
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Segunda división: train vs val
    val_ratio = val_size / train_size
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, 
        random_state=random_state, stratify=y_train_val
    )
    
    print(f"Dataset: Wine Classification")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    training_set = TensorDataset(X_train, y_train)
    validation_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    datasets = [training_set, validation_set, test_set]
    
    return datasets

def load_and_log():
    # 🚀 start a run with wandb
    with wandb.init(
        project="MLOps-Pycon2023",
        name="Load Raw Data Wine", 
        job_type="load-data") as run:
        
        datasets = load()
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            "wine-raw", type="dataset",
            description="raw Wine dataset from sklearn, split into train/val/test",
            metadata={
                "source": "sklearn.datasets.load_wine",
                "sizes": [len(dataset) for dataset in datasets],
                "n_features": 13,
                "n_classes": 3,
                "task": "multiclass classification"
            })

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B
        run.log_artifact(raw_data)

if __name__ == "__main__":
    load_and_log()
