import torch
from torch.utils.data import TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
# Testing
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
parser.add_argument('--dataset', type=str, default='digits', 
                    help='Dataset name from sklearn (digits, iris, wine, breast_cancer)')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def load(train_size=0.8, val_size=0.1, random_state=42):
    """
    Load data from scikit-learn datasets
    """
    
    # Diccionario de datasets disponibles en sklearn
    dataset_loaders = {
        'digits': load_digits,
        'iris': __import__('sklearn.datasets', fromlist=['load_iris']).load_iris,
        'wine': __import__('sklearn.datasets', fromlist=['load_wine']).load_wine,
        'breast_cancer': __import__('sklearn.datasets', fromlist=['load_breast_cancer']).load_breast_cancer,
    }
    
    # Cargar el dataset seleccionado
    dataset_name = args.dataset if hasattr(args, 'dataset') else 'digits'
    loader = dataset_loaders.get(dataset_name, load_digits)
    data = loader()
    
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
    
    print(f"Dataset: {dataset_name}")
    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(torch.unique(y))}")
    
    training_set = TensorDataset(X_train, y_train)
    validation_set = TensorDataset(X_val, y_val)
    test_set = TensorDataset(X_test, y_test)
    datasets = [training_set, validation_set, test_set]
    
    return datasets, dataset_name, X.shape[1], len(torch.unique(y))

def load_and_log():
    # 🚀 start a run, with a type to label it and a project it can call home through wandb
    dataset_name = args.dataset if hasattr(args, 'dataset') else 'digits'
    
    with wandb.init(
        project="MLOps-Pycon2023",
        name=f"Load Raw Data {dataset_name} ExecId-{args.IdExecution}", 
        job_type="load-data") as run:
        
        datasets, ds_name, n_features, n_classes = load()
        names = ["training", "validation", "test"]

        # 🏺 create our Artifact
        raw_data = wandb.Artifact(
            f"{ds_name}-raw", type="dataset",
            description=f"raw {ds_name} dataset from sklearn, split into train/val/test",
            metadata={
                "source": f"sklearn.datasets.load_{ds_name}",
                "sizes": [len(dataset) for dataset in datasets],
                "n_features": n_features,
                "n_classes": n_classes
            })

        for name, data in zip(names, datasets):
            # 🐣 Store a new file in the artifact, and write something into its contents.
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        # ✍️ Save the artifact to W&B.
        run.log_artifact(raw_data)

# testing
if __name__ == "__main__":
    load_and_log()
