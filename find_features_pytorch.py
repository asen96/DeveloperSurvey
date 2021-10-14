import os 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import train

data_path = '../data/columns/'

# Function to find the best 15 features to train a NN to predict annual salary
def best_features(path:str):

    # Define an empty dictionary to store losses
    loss_dict = dict()

    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
    
    folder_path = f'{data_path}{folder}/'
    for file in os.listdir(folder_path):
        
        if file == '.DS_Store':
            continue
        
        # Each feature has an individual csv file, which contains the target salaries
        if file.endswith('.csv'):
            
            print(file)
            feature_dict = dict()
            file_path = f'{folder_path}{file}'
            df = pd.read_csv(file_path).fillna(0.0)
            
            # Drop columns which have no responses
            to_drop = [x for x in list(df.columns) if df[x].sum() == 0]
            df = df.drop(columns=to_drop)
            columns = list(df.columns)
            df = train.scale(df, columns)
                
            feature, target = train.get_feature_target(df)
            X_train, X_val, y_train, y_val = train_test_split(feature, target, test_size=0.2)
            input_layers = X_train.shape[1]
            output_layers = 1
            first_layer_nodes = 1
            
            training_loader, validation_loader = train.get_loader(X_train, X_val, y_train, y_val)
            
            network = train.FFNetwork(input_layers, output_layers, first_layer_nodes)
            optimizer = train.torch.optim.Adam(network.parameters(), lr=0.007)
            loss_fn = train.nn.MSELoss()
            epochs = 1
            
            train_losses, epoch_list, val_losses = train.train(network, training_loader, validation_loader, optimizer, loss_fn, epochs)
            
            # Save the network, losses in a dictionary, the keys are the features
            key = file.replace('.csv', '')
            feature_dict['network'] = network
            feature_dict['training_loss'] = train_losses
            feature_dict['validation_loss'] = val_losses
            feature_dict['epoch_list'] = epoch_list
            loss_dict[key] = feature_dict

    # Sort according to validation loss and get the top 15 features
    val_loss = []
    for key in dict.keys(loss_dict):
        val_loss.append((key, np.mean(loss_dict[key]['validation_loss'])))

    val_loss_dict = dict()
    for i in val_loss:
        val_loss_dict[i[0]] = i[1]

    val_loss_dict = dict(sorted(val_loss_dict.items(), key=lambda item: item[1]))
    top_15 = [list(dict.keys(val_loss_dict))[i] for i in range(0,16)]
    top_15 = [x for x in top_15 if x != 'ConvertedComp']

    return top_15

# Function to take the 15 best features and save them in one csv file, to make learning easier for the final network
def make_csv(features):
    df_15 = pd.DataFrame()
    flag = 0
    for folder in os.listdir(data_path):
        if folder == '.DS_Store':
            continue
    
        folder_path = f'{data_path}{folder}/'
        for file in os.listdir(folder_path):
        
            if file == '.DS_Store':
                continue
                
            if file.endswith('.csv') and file.replace('.csv','') in features:
                    
                print(file)
                file_path = f'{folder_path}{file}'
                df = pd.read_csv(file_path).fillna(0.0)
                if flag == 1:
                    df = df.drop(columns='ConvertedComp')
                    if 'Other(s):' in list(df.columns):
                        df = df.drop(columns='Other(s):')
                    
                if file == 'CurrencySymbol.csv':
                    df = df.rename(columns={'PHP':'PHP_currsym'})
                    
                to_drop = [x for x in list(df.columns) if df[x].sum() == 0]
                df = df.drop(columns=to_drop)
                df= df.drop(columns='Respondent')
                columns = list(df.columns)
                df = train.scale(df, columns)
                df_15[columns] = df[columns]
                flag = 1

    df_15.to_csv(f'{data_path}/best_features.csv')

def main():
    features = best_features(data_path)
    make_csv(features)

if __name__==main:
    main()