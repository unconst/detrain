from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from tqdm import tqdm
import pandas as pd
import random
import itertools
from torchvision import datasets, transforms

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
infinite_train_loader = itertools.cycle(train_loader)

# Load the MNIST validation dataset
val_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

def evaluate_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()  # Set the model to evaluation mode
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

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def join(modelA: torch.nn.Module, modelB: torch.nn.Module) -> None:
    # Get the state dictionaries of the two models
    state_dict_A = modelA.state_dict()
    state_dict_B = modelB.state_dict()
    
    # Compute the average of the parameters directly using dictionary comprehension
    averaged_state_dict = {key: (state_dict_A[key] + state_dict_B[key]) / 2 
                           for key in state_dict_A if key in state_dict_B}
    
    # Load the updated state dictionaries back into the models
    modelA.load_state_dict(averaged_state_dict)
    modelB.load_state_dict(averaged_state_dict)

num_epochs = 1
max_batches = 10000
batches_per_eval = 100
num_nodes = 10
join_prob = 0.1
batches_per_join = 100

nodes = []
for i in range( num_nodes ):
    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    nodes.append( (model, optimizer ) )

history_df = pd.DataFrame(columns=['batch', 'base', 'max', 'min', 'mean', 'maxwin', 'minwin', 'meanwin'])

n_joins = 0
n_batches = 0
while True:
        # Run optimizer over models
        for i, (model, optimizer ) in enumerate(nodes):
            images, labels = next( infinite_train_loader )
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Increment number of batches seen
        n_batches += 1

        # Join models every batches_per_join batches
        if n_batches % batches_per_join == 0:
            midx_1 = list(range(1, num_nodes))
            midx_2 = list(range(1, num_nodes))
            random.shuffle(midx_1); random.shuffle(midx_2)
            for i in midx_1:
                for j in midx_2:
                    if i != j:
                        if random.random() < join_prob:
                            join(nodes[i][0], nodes[j][0])
                            n_joins += 1

        if n_batches % batches_per_eval == 0:
            results = []
            for i, (model, _) in enumerate(nodes):
                accuracy, avg_loss = evaluate_model(model, val_loader, criterion, device)
                results.append( accuracy )

            # Append to the DataFrame using pandas.concat
            base = results[0]
            max_val = max(results[1:])
            min_val = min(results[1:])
            mean_val = sum(results[1:])/(num_nodes-1)
            maxwin = max_val - base
            minwin = min_val - base
            meanwin = mean_val - base
            new_row = pd.DataFrame({
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
            history_df = pd.concat([history_df, new_row], ignore_index=True)
            print(f"Batch {n_batches}, Joins {n_joins}, Base {base}, Max: {max_val}, Min: {min_val}, Mean: {mean_val}, MaxWin: {maxwin}, MeanWin: {meanwin} MinWin: {minwin} ")
            history_df.to_csv('history.csv', index=False)

        if n_batches >= max_batches:
            break