import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm
import train 

df = pd.read_csv('../data/columns/best_features.csv')

# Train the network on the fifteen features chosen before and use k-fold cross-validation
def main():

    folds = 10
    length = len(df)
    size = int(length / folds+1)
    train_set = list(range(0, length, size))

    fold_loss = []
    networks = []
    columns = []

    for lower_bound in train_set:
        higher_bound = min(lower_bound + size, length)
        temp = df.copy(True)
        test_df = temp.isin(range(lower_bound, higher_bound))
        training_df = temp.drop(range(lower_bound, higher_bound), inplace=False)
        
        X_train, y_train = shuffle(training_df.drop(columns='ConvertedComp'), training_df['ConvertedComp'])
        X_val, y_val = shuffle(test_df.drop(columns='ConvertedComp'), test_df['ConvertedComp'])
        input_layers = X_train.shape[1]
        output_layers = 1
        first_layer_nodes = 1
        
        training_loader, validation_loader = train.get_loader(X_train.values, X_val.values, y_train.values, y_val.values, 1)
        
        network = train.FFNetwork(input_layers, output_layers, first_layer_nodes)
        optimizer = train.torch.optim.Adam(network.parameters(), lr=0.001)
        loss_fn = train.nn.MSELoss()
        epochs = 1
        
        train_losses, epoch_list, val_losses = train.train(network, training_loader, validation_loader, optimizer, loss_fn, epochs)
        
        fold_loss.append(val_losses)
        networks.append(network)
        columns.append(list(X_train.columns))

    results = pd.DataFrame(columns=['Fold','Val_loss_epoch_1'])
    for index, value in enumerate(fold_loss):
        results.loc[index,:] = [index, value[0]]
    
    results.to_csv('../data/results.csv')
    
if __name__ == 'main':
    main()