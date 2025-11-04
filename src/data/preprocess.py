import torch
from torch.utils.data import TensorDataset
from sklearn.preprocessing import StandardScaler
import os
import wandb


def preprocess(dataset, normalize=True, scale_features=True):
    """
    ## Prepare the wine data
    """
    x, y = dataset.tensors

    if normalize:
        # Normalize features using StandardScaler (mejor para datos tabulares)
        scaler = StandardScaler()
        x_numpy = x.numpy()
        x_scaled = scaler.fit_transform(x_numpy)
        x = torch.FloatTensor(x_scaled)
    
    # No necesitamos expand_dims porque no son imágenes
    
    return TensorDataset(x, y)


def preprocess_and_log(steps):

    with wandb.init(
        project="MLOps-Pycon2023",
        name="Preprocess Wine Data", 
        job_type="preprocess-data") as run:    
        
        processed_data = wandb.Artifact(
            "wine-preprocess", type="dataset",
            description="Preprocessed Wine dataset with standardization",
            metadata=steps)
         
        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('wine-raw:latest')

        # 📥 if need be, download the artifact
        raw_dataset = raw_data_artifact.download(root="./data/artifacts/")
        
        for split in ["training", "validation", "test"]:
            raw_split = read(raw_dataset, split)
            processed_dataset = preprocess(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)


def read(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))

    return TensorDataset(x, y)


# Steps adaptados para datos tabulares
steps = {
    "normalize": True,      # StandardScaler para normalizar features
    "scale_features": True  # Mantener coherencia con normalize
}


if __name__ == "__main__":
    preprocess_and_log(steps)
