import torch
import torch.nn.functional as F
from torch import nn 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Import the model class
from src.Classifier import WineClassifier

import os
import wandb


# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)


def read(data_dir, split):
    """
    Read data from a directory and return a TensorDataset object.

    Args:
    - data_dir (str): The directory where the data is stored.
    - split (str): The name of the split to read (e.g. "training", "validation", "test").

    Returns:
    - dataset (TensorDataset): A TensorDataset object containing the data.
    """
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)


def train(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    model.train()
    example_ct = 0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)
            epoch_loss += loss.item()

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))
                
                train_log(loss, example_ct, epoch)

        # Evaluate the model on the validation set at each epoch
        val_loss, val_accuracy = test(model, valid_loader)  
        test_log(val_loss, val_accuracy, example_ct, epoch)
        
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

    
def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def train_log(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
    

def test_log(loss, accuracy, example_ct, epoch):
    loss = float(loss)
    accuracy = float(accuracy)
    wandb.log({"epoch": epoch, "validation/loss": loss, "validation/accuracy": accuracy}, step=example_ct)
    print(f"Loss/accuracy after {str(example_ct).zfill(5)} examples: {loss:.3f}/{accuracy:.2f}%")


def evaluate(model, test_loader):
    """
    Evaluate the trained model and get metrics + confusion matrix data
    """
    loss, accuracy = test(model, test_loader)
    highest_losses, hardest_examples, true_labels, predictions = get_hardest_k_examples(
        model, test_loader.dataset, k=10  # Menos ejemplos porque el dataset es pequeño
    )

    return loss, accuracy, highest_losses, hardest_examples, true_labels, predictions


def get_hardest_k_examples(model, testing_set, k=10):
    """
    Get the k examples with highest loss (hardest to classify)
    """
    model.eval()
    loader = DataLoader(testing_set, 1, shuffle=False)

    losses = []
    predictions = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=1, keepdim=True)
            
            losses.append(loss.item())
            predictions.append(pred.item())

    losses = torch.tensor(losses)
    predictions = torch.tensor(predictions)
    
    argsort_loss = torch.argsort(losses, dim=0)
    
    # Get the k hardest examples
    highest_k_losses = losses[argsort_loss[-k:]]
    hardest_k_examples = testing_set[argsort_loss[-k:]][0]
    true_labels = testing_set[argsort_loss[-k:]][1]
    predicted_labels = predictions[argsort_loss[-k:]]

    return highest_k_losses, hardest_k_examples, true_labels, predicted_labels


def train_and_log(config, experiment_id='001'):
    with wandb.init(
        project="MLOps-Pycon2023", 
        name=f"Train Wine Model Experiment-{experiment_id}", 
        job_type="train-model", 
        config=config) as run:
        
        config = wandb.config
        
        # Download preprocessed data
        data = run.use_artifact('wine-preprocess:latest')
        data_dir = data.download()

        training_dataset = read(data_dir, "training")
        validation_dataset = read(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
        
        # Load initialized model
        model_artifact = run.use_artifact("WineClassifier:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_model_WineClassifier.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        model = WineClassifier(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        
        print(f"\n{'='*50}")
        print(f"Training Configuration - Experiment {experiment_id}")
        print(f"{'='*50}")
        print(f"Epochs: {config.epochs}")
        print(f"Batch Size: {config.batch_size}")
        print(f"Learning Rate: {config.learning_rate}")
        print(f"Optimizer: {config.optimizer}")
        print(f"{'='*50}\n")
 
        train(model, train_loader, validation_loader, config)

        # Save trained model
        model_artifact = wandb.Artifact(
            f"trained-wine-model-exp{experiment_id}", 
            type="model",
            description=f"Trained Wine Classifier - Experiment {experiment_id}",
            metadata=dict(model_config))

        torch.save(model.state_dict(), f"trained_model_exp{experiment_id}.pth")
        model_artifact.add_file(f"trained_model_exp{experiment_id}.pth")
        wandb.save(f"trained_model_exp{experiment_id}.pth")

        run.log_artifact(model_artifact)

    return model

    
def evaluate_and_log(experiment_id='001', config=None):
    
    with wandb.init(
        project="MLOps-Pycon2023", 
        name=f"Eval Wine Model Experiment-{experiment_id}", 
        job_type="eval-model", 
        config=config) as run:
        
        # Download preprocessed data
        data = run.use_artifact('wine-preprocess:latest')
        data_dir = data.download()
        testing_set = read(data_dir, "test")

        test_loader = DataLoader(testing_set, batch_size=32, shuffle=False)

        # Load trained model
        model_artifact = run.use_artifact(f"trained-wine-model-exp{experiment_id}:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, f"trained_model_exp{experiment_id}.pth")
        model_config = model_artifact.metadata

        model = WineClassifier(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        loss, accuracy, highest_losses, hardest_examples, true_labels, preds = evaluate(model, test_loader)

        run.summary.update({"test/loss": loss, "test/accuracy": accuracy})
        
        print(f"\n{'='*50}")
        print(f"Test Results - Experiment {experiment_id}")
        print(f"{'='*50}")
        print(f"Test Loss: {loss:.4f}")
        print(f"Test Accuracy: {accuracy:.2f}%")
        print(f"{'='*50}\n")

        # Log hardest examples (tabular data, no images)
        hardest_table = wandb.Table(
            columns=["example_id", "true_label", "predicted_label", "loss"],
            data=[
                [i, int(true_label), int(pred), float(loss_val)] 
                for i, (true_label, pred, loss_val) in enumerate(zip(true_labels, preds, highest_losses))
            ]
        )
        
        wandb.log({"hardest_examples": hardest_table})


if __name__ == "__main__":
    # Diferentes configuraciones de experimentos
    experiments = [
        {
            "id": "001",
            "config": {
                "batch_size": 16,
                "epochs": 100,
                "batch_log_interval": 5,
                "optimizer": "Adam",
                "learning_rate": 0.001
            }
        },
        {
            "id": "002",
            "config": {
                "batch_size": 16,
                "epochs": 150,
                "batch_log_interval": 5,
                "optimizer": "Adam",
                "learning_rate": 0.0005
            }
        },
        {
            "id": "003",
            "config": {
                "batch_size": 32,
                "epochs": 200,
                "batch_log_interval": 3,
                "optimizer": "SGD",
                "learning_rate": 0.01
            }
        }
    ]
    
    # Ejecutar experimentos
    for exp in experiments:
        print(f"\n{'#'*60}")
        print(f"Starting Experiment {exp['id']}")
        print(f"{'#'*60}\n")
        
        model = train_and_log(exp['config'], experiment_id=exp['id'])
        evaluate_and_log(experiment_id=exp['id'], config=exp['config'])
