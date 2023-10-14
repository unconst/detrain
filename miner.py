# Description: This script trains an ensemble of models on the MNIST dataset.
# The models are trained in parallel and independently, and their parameters are averaged periodically.
# The script tracks the performance of the ensemble over time, and saves the results to a CSV file.
import torch
import random
import itertools
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from typing import Any, Tuple
from torchvision import datasets, transforms
from rich.console import Console
console = Console()

# Check if GPU is available for computations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the training and validation datasets
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
infinite_train_loader = itertools.cycle(train_loader)
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluates the model on a given dataloader.
# Returns:
# - accuracy: Model accuracy on the dataset
# - average_loss: Average loss on the dataset
def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += len(labels)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return accuracy, average_loss

# Define a simple feed-forward neural network for MNIST
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()  # Flattens the input tensor
        self.fc1 = nn.Linear(28*28, 128)  # First dense layer
        self.fc2 = nn.Linear(128, 10)     # Second dense layer leading to 10 classes

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Joins the parameters of two models by averaging them.
# Both models will be updated with the averaged parameters.
# Arguments:
# - modelA: The first model to be averaged
# - modelB: The second model to be averaged
# Returns: Models are updated in place. No return value.
def join(modelA: torch.nn.Module, modelB: torch.nn.Module) -> None:
    # Extract the state dictionaries containing parameters of each model
    state_dict_A = modelA.state_dict()
    state_dict_B = modelB.state_dict()
    
    # Calculate the average of the parameters
    averaged_state_dict = {key: (state_dict_A[key] + state_dict_B[key]) / 2 for key in state_dict_A}

    # Update both models with the averaged parameters
    modelA.load_state_dict(averaged_state_dict)
    modelB.load_state_dict(averaged_state_dict)

# max_batches: The total number of mini-batches that will be processed during training.
# This hyperparameter sets an upper limit for the training iterations, controlling the 
# duration of training and determining when to stop.
max_batches = 10000

# batches_per_eval: Specifies how frequently the models should be evaluated.
# After every 'batches_per_eval' mini-batches processed, the models' performance is 
# assessed on a validation set. Lower values lead to more frequent evaluations, but 
# may increase the computational overhead.
batches_per_eval = 1000

# num_nodes: The number of individual model instances or "nodes" in the ensemble.
# Each node is a separate model instance that is trained and potentially averaged 
# with others. This parameter determines the diversity and size of the ensemble.
num_nodes = 5

# join_prob: The probability that a pair of models (or nodes) will have their parameters 
# averaged (or "joined") during the joining phase. A higher value increases the likelihood 
# of averaging, promoting parameter convergence across nodes.
join_prob = 0.1  # Probability with which two models are averaged

# batches_per_join: Defines the frequency at which the model parameters may be averaged.
# After every 'batches_per_join' mini-batches, a random check based on 'join_prob' is made 
# to decide if two models should be joined. It provides a structured interval for potential joining.
batches_per_join = 1000

# Initialize the 'nodes' list. In the context of this training pipeline, each node represents an individual model-optimizer pair.
# 1. An instance of the 'Net' model, which has been transferred to the specified computing device (either CPU or GPU).
# 2. An Adam optimizer that is set up to optimize the parameters of the associated 'Net' model with a learning rate of 0.001.
# The list 'nodes' will have 'num_nodes' such pairs, allowing for parallel and independent training of multiple model instances.
nodes = []
for _ in range(num_nodes):
    model = Net().to(device); optimizer = optim.Adam(model.parameters(), lr=0.001)
    nodes.append((model, optimizer))

# To track the progress and results.
# We record the following metrics:
# - batch: The number of mini-batches processed during training.
# - n_joins: The number of times the model parameters have been averaged.
# - base: The accuracy of the first model in the ensemble.
# - max: The accuracy of the best model in the ensemble.
# - min: The accuracy of the worst model in the ensemble.
# - mean: The average accuracy of the ensemble.
# - maxwin: The difference between the base value and the best model's accuracy.
# - minwin: The difference between the base value and the worst model's accuracy.
# - meanwin: The difference between the base value and the average accuracy of the ensemble.
history_df = pd.DataFrame(columns=['batch', 'n_joins', 'base', 'max', 'min', 'mean', 'maxwin', 'minwin', 'meanwin'])

# Training loop, which runs until the maximum number of mini-batches has been processed.
n_joins = 0
n_batches = 0
while n_batches < max_batches:

    # Trains each model on their next mini-batch of data.
    # the dataset is infinite, so we can just keep looping over it.
    # each model is trained using its own optimizer and gets a unique mini-batch of data.
    for model, optimizer in nodes:
        images, labels = next(infinite_train_loader)
        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

    n_batches += 1

    # Joins models periodically by averaging parameters between pairs of models. 
    # The joining operation is not performed every batch but rather at intervals defined by 'batches_per_join'.
    # For all possible pairs of models we:
    # - ensure they are distinct pairs
    # - use a random probability check to decide if the parameters of the current pair of models should be averaged based on join_prob.
    # - If the conditions are met, the 'join' function is invoked to average the parameters of the two selected models.
    if n_batches % batches_per_join == 0:
        pairs = list(itertools.product(range(num_nodes), repeat=2))
        random.shuffle(pairs)
        for i, j in pairs:
            if i != j and random.random() < join_prob:
                join(nodes[i][0], nodes[j][0])
                n_joins += 1

    # Evaluate and log metrics periodically.
    # The evaluation is performed every 'batches_per_eval' mini-batches.
    # The models are evaluated on a validation set, and the results are logged to a CSV file.
    if n_batches % batches_per_eval == 0:
        # Eval each model on the validation set.
        results = [evaluate_model(model, val_loader, nn.CrossEntropyLoss(), device)[0] for model, _ in nodes]

        # Calculate the base value and the max, min, and mean values for the results.
        # The base value is the accuracy of the first model in the ensemble.
        # The max, min, and mean values are calculated from the remaining models.
        # We also measure the difference between the base value and the max, min, and mean values.
        # These values are used to track the performance of the ensemble.
        base = results[0]
        max_val = max(results[1:])
        min_val = min(results[1:])
        mean_val = sum(results[1:]) / (num_nodes - 1)
        maxwin = max_val - base
        minwin = min_val - base
        meanwin = mean_val - base

        # Append the results to the history dataframe
        # Print the output to the console
        # Save the latests to the CSV file.
        history_df = pd.concat([history_df, 
            pd.DataFrame({
                'batch': [n_batches],
                'n_joins': [n_joins],
                'base': [base],
                'max': [max_val],
                'min': [min_val],
                'mean': [mean_val],
                'maxwin': [maxwin],
                'minwin': [minwin],
                'meanwin': [meanwin],
            })
        ], ignore_index=True)
        output = f"""Batch: {n_batches}, Joins: {n_joins}, Base: [bold blue]{base:.4f}[/], Max: {max_val:.4f}, Min: {min_val:.4f}, Mean: {mean_val:.4f}, MaxWin: [bold {"green" if maxwin > 0 else "red"}]{maxwin:.4f}[/], MeanWin: [bold {"green" if meanwin > 0 else "red"}]{meanwin:.4f}[/], MinWin: [bold {"green" if minwin > 0 else "red"}]{minwin:.4f}[/]"""
        console.print(output)
        history_df.to_csv('history.csv', index=False)

