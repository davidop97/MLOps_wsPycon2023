import torch
import os
import wandb

# Import the model class
from WineClassifier import WineClassifier


# Check if the directory "./model" exists
if not os.path.exists("./model"):
    os.makedirs("./model")


# Data parameters para Wine dataset
num_classes = 3        # 3 tipos de vinos
input_shape = 13       # 13 características químicas


def build_model_and_log(config, model, model_name="WineClassifier", model_description="MLP for Wine Classification"):
    with wandb.init(
        project="MLOps-Pycon2023", 
        name="Initialize Wine Model", 
        job_type="initialize-model", 
        config=config) as run:
        
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_model_{model_name}.pth"

        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        # ➕ another way to add a file to an Artifact
        model_artifact.add_file(f"./model/{name_artifact_model}")

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)
        
        print(f"Model saved: {name_artifact_model}")
        print(f"Model architecture:\n{model}")


# Configuración del modelo para Wine dataset
model_config = {
    "input_shape": input_shape,      # 13 features
    "hidden_layer_1": 64,            # Primera capa oculta (más grande para capturar patrones)
    "hidden_layer_2": 32,            # Segunda capa oculta
    "num_classes": num_classes,      # 3 clases de vinos
    "dropout": 0.3                   # Regularización para evitar overfitting
}


# Crear el modelo
model = WineClassifier(**model_config)


# Build and log model
if __name__ == "__main__":
    build_model_and_log(
        model_config, 
        model, 
        "WineClassifier",
        "MLP Classifier for Wine Dataset with BatchNorm and Dropout"
    )
