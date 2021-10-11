import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import os
import csv

def build_and_compile_model():
    model = keras.Sequential([
        layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),
                  tf.keras.metrics.MeanSquaredError(),
                  tf.keras.metrics.MeanAbsoluteError()])
    return model

def train_on_df(df):

    df, target = clean(df)
    X_train, X_test, y_train, y_test = train_test_split(df, target,
                                                        test_size=0.2)

    model = build_and_compile_model()

    history = model.fit(X_train.values, y_train.values,
                        verbose=0,
                        epochs=1, batch_size=1,
                        validation_data=(X_test, y_test))
    return model, history, list(df.columns)


def scale(df:pd.DataFrame, columns:list) -> (pd.DataFrame):

    for column in columns:

        if column == 'Respondent':
            continue

        mean = df[column].mean()
        dev = 3*df[column].std()
        scale = 1./(mean + dev)
        df[column] = df[column] * scale

    return df

def get_feature_target(df:pd.DataFrame) -> (np.ndarray, np.ndarray):

    target = df['ConvertedComp']
    feature = df.drop(columns=['ConvertedComp','Respondent'])

    return feature.values, target.values

loss_dict = dict()
data_path = './data/columns/'


with open('./data/metrics.csv', 'w') as csv_file:

    writer = csv.DictWriter(csv_file, fieldnames=['MSE','RMSE'])
    writer.writeheader()

    for folder in os.listdir(data_path):

        if folder != 'numeric':
            continue

        folder_path = f'{data_path}{folder}/'
        for file in os.listdir(folder_path):

            if file == '.DS_Store':
                continue

            if file.endswith('.csv'):

                print(file)
                feature_dict = dict()
                file_path = f'{folder_path}{file}'
                df = pd.read_csv(file_path)


                columns = list(df.columns)
                df = scale(df, columns)

                feature, target = get_feature_target(df)
                X_train, X_val, y_train, y_val = train_test_split(feature, target, test_size=0.2)

                model = build_and_compile_model()

                history = model.fit(X_train, y_train, verbose=1, epochs=1, batch_size=1,
                                    validation_data=(X_val, y_val))


                key = file.replace('.csv', '')
                #feature_dict['model'] = model
                feature_dict['RMSE'] = history.history['root_mean_squared_error'][0]
                feature_dict['MSE'] = history.history['mean_squared_error'][0]
                #loss_dict[key] = feature_dict
                writer.writerow(feature_dict)
