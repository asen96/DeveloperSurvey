import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Define the neural network
class FFNetwork(nn.Module):
    
    def __init__(self,input_layers,output_layers,first_layer_nodes):
        super(FFNetwork, self).__init__()
        self.first_layer = nn.Linear(input_layers,first_layer_nodes)
        self.output_layer = nn.Linear(first_layer_nodes,output_layers)
        
    def forward(self, x):
        x = self.first_layer(x)
        x = self.output_layer(x)
        return x

# Training function
def train(network, train_set, val_set, optim, loss_fn, epochs):
    
    epoch_values = []
    val_losses = []
    train_losses = []
    
    for epoch in tqdm(range(epochs)):
        
        epoch_losses = []
        network.train()
        for train_data, train_target in train_set:
            optim.zero_grad()
            target = network(train_data)
            loss = loss_fn(train_target, target)
            loss.backward()
            optim.step()
            epoch_losses.append(loss.item())
                    
        train_losses.append(np.mean(epoch_losses))
        
        if epoch % 1 == 0:
            epoch_losses = []
            network.eval()
            for val_data, val_target in val_set:
                target = network(val_data)
                loss = loss_fn(val_target, target)
                epoch_losses.append(loss.detach().numpy())
            
            val_losses.append(np.mean(epoch_losses))
            epoch_values.append(epoch)
            
    return train_losses, epoch_values, val_losses

# Function to prepare the data in torch DataLoaders
def get_loader(X_train, X_val, y_train, y_val, batch_size):
    
    training_set = torch.Tensor(X_train)
    training_target = torch.Tensor(y_train)
    
    validation_set = torch.Tensor(X_val)
    validation_target = torch.Tensor(y_val)
    
    training_dataset = torch.utils.data.TensorDataset(training_set, training_target)
    validation_dataset = torch.utils.data.TensorDataset(validation_set, validation_target)
    
    training_loader = DataLoader(training_dataset, batch_size=batch_size, drop_last=False)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, drop_last=False)
    
    return training_loader, validation_loader

# Function to scale the columns by dividing by the upper limit, defined as 3*sigma beyond the mean
def scale(df:pd.DataFrame, columns:list):
    
    for column in columns:
        
        if column == 'Respondent':
            continue
            
        mean = df[column].mean() if df[column].mean() != 0.0 else 1
        dev = 3*df[column].std() if df[column].std() != 0.0 else 1
        scale = 1./(mean + dev)
        df[column] = df[column] * scale
        
    return df

# Function extracts the feature and target columns from the data frame
def get_feature_target(df:pd.DataFrame):
    
    target = df['ConvertedComp']
    feature = df.drop(columns=['ConvertedComp'])
    if 'Respondent' in list(df.columns):
        feature = feature.drop(columns='Respondent')
    return feature.values, target.values