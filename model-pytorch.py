import pandas as pd
import numpy as np
import math
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

sns.set_context("notebook")
rng = np.random.default_rng(seed=2021)

df = pd.read_csv('./data/clean_data.csv')
reduced_df = pd.read_csv('./data/reduced_categorized.csv')
df.set_index('Respondent', inplace = True)
reduced_df.set_index('Respondent', inplace = True)

#Define the NN in PyTorch
class FFNetwork(nn.Module):
    def __init__(self, input_size, output_size, first_layer_nodes, second_layer_nodes):
        super(FFNetwork, self).__init__()
        self.first_layer = nn.Linear(input_size, first_layer_nodes)
        self.second_layer = nn.Linear(first_layer_nodes, second_layer_nodes)
        self.output_layer = nn.Linear(second_layer_nodes, output_size)

    def forward(self, x):
        x = self.first_layer(x)
        x = nn.functional.dropout(x, 0.5)
        x = nn.functional.relu(x)
        x = self.second_layer(x)
        x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x

#Define the training loop, validating the model every 20 epochs
def train(network, train_set, val_set, optim, loss_fn, epochs):
    epoch_values = []
    val_losses = []
    train_losses = []
    train_accuracy = []
    val_accuracy = []

    for epoch in tqdm(range(epochs)):

        training_loss = 0.0
        training_count = 0
        batch_accuracy = []
        train_pred = []
        val_pred = []
        network.train()
        for train_data, train_labels in train_set:
            train_pred = []
            optim.zero_grad()
            labels = network(train_data)
            train_pred.append(labels)
            loss = loss_fn(train_labels, labels[:,0:6])
            loss.backward()
            optim.step()
            training_loss += loss.item()

            batch_accuracy.append(accuracy(train_labels, train_pred, False))

        train_losses.append(training_loss)
        train_accuracy.append(np.mean(batch_accuracy))

        if epoch % 20 == 0:
            epoch_losses = []
            epoch_accuracy = []
            val_count = 0
            network.eval()
            for val_data, val_labels in val_set:
                labels = network(val_data)
                val_pred.append(labels)
                loss = loss_fn(val_labels, labels[:, 0:6])
                epoch_losses.append(loss.detach().numpy())

                epoch_accuracy.append(accuracy(val_labels, val_pred, False))

            val_losses.append(np.mean(epoch_losses))
            val_accuracy.append(np.mean(epoch_accuracy))
            epoch_values.append(epoch)

    return train_losses, epoch_values, val_losses, train_accuracy, val_accuracy

#Define test function, testing the model on the test set
def test(network, test_set, actual_labels, loss_fn):
    test_losses = []
    predictions = []
    #test_count = 0
    batch_accuracy = []
    accuracy_labels = torch.Tensor

    network.eval()
    for test_data, test_labels in tqdm(test_set):
        labels = network.forward(test_data)
        predictions.append(labels)
        loss = loss_fn(test_labels, labels[:, 0:6])
        test_losses.append(loss.detach().numpy())

    test_accuracy = accuracy(actual_labels, predictions, True)

    return test_losses, test_accuracy, predictions

#Define method to calculate accuracy
def accuracy(actual_labels, predictions, test_bool):
    actual_labels = actual_labels.tolist()
    if test_bool == False:
        predictions = predictions[0].tolist()
    entries = len(predictions)
    correct = 0

    if test_bool:
        for i in range(entries):
            predictions[i] = predictions[i].tolist()

    predictions, actual_labels = np.vstack(predictions), np.vstack(actual_labels)

    for i in range(entries):
        if sum([x.round()==y for (x,y) in zip(predictions[i], actual_labels[i])]) == 6.0:
            correct += 1

    return correct/len(predictions)


data = reduced_df.values.astype(np.int64)

shuffled_indices = rng.permutation(data.shape[0])

#Preprocess features and target to feed into the PyTorch training loop
training_set = torch.Tensor(data[shuffled_indices[:40000],6:100])
training_labels = torch.Tensor(data[shuffled_indices[:40000],0:6])

validation_set = torch.Tensor(data[shuffled_indices[40000:45000],6:100])
validation_labels = torch.Tensor(data[shuffled_indices[40000:45000],0:6])

test_set = torch.Tensor(data[shuffled_indices[45000:],6:100])
test_labels = torch.Tensor(data[shuffled_indices[45000:],0:6])

training_dataset = torch.utils.data.TensorDataset(training_set, training_labels)
validation_dataset = torch.utils.data.TensorDataset(validation_set, validation_labels)
test_dataset = torch.utils.data.TensorDataset(test_set, test_labels)

training_loader = DataLoader(training_dataset, batch_size=100, drop_last=False)
validation_loader = DataLoader(validation_dataset, batch_size=100, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=1, drop_last=False)

# Define an instance of the network, with 94 input features, 6 output categories,
# 64 neurons in the first layer, and 32 in the last layer.
# Using the Adam optimizer, mean-squared-error as the loss function and 1000 learning epochs
net = FFNetwork(94, 6, 64, 32)
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
epochs = 1000

# Call the training function
train_losses, epoch_list, val_losses, train_accuracy, val_accuracy = train(net, training_loader, validation_loader, optimizer, loss_fn, epochs)

#Plot the loss and accuracy of the network
plt.rcParams['figure.figsize'] = [10, 10]
fig, (ax1, ax2) = plt.subplots(2,1,sharex=True)
sns.lineplot(x=epoch_list, y=val_losses, ax=ax1, label='validation loss').set_title('lr=0.01, epochs=1000')
sns.lineplot(x=np.arange(epochs), y=train_losses, ax=ax1, label='training loss')
sns.lineplot(x=epoch_list, y=val_accuracy, ax=ax2, label='validation accuracy')
sns.lineplot(x=np.arange(epochs), y=train_accuracy, ax=ax2, label='training accuracy')
