import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import math
import re
from tqdm import tqdm
import os

# Function that takes an exclusive column (one where the respondent can only select one answer, eg. country, employment status)
# and explodes it into one hot encoded columns, for each possible response. The salary column (ConvertedComp) is appended to 
# make learning easier while training later.
def exc_col(column:str) -> (pd.DataFrame):
    
    df_expanded = (pd.get_dummies(df[column])).join(df['ConvertedComp'])
    return df_expanded

# For columns which allow multiple responses (eg. programming languages used), each response is split with the ';' delimiter, 
# and for each unique response, a one hot encoded column is created.
def mul_col(column:str) -> (pd.DataFrame):
    
    entry = dict()
    df_expanded = pd.DataFrame()
    for i in range(1,len(df)+1):
        
        entry = dict()
        if type(df[column][i]) == float and math.isnan(df[column][i]):
            # Column for NA response
            entry[column+'_na'] = np.int64(1)
            df_expanded = df_expanded.append(entry, ignore_index=True).fillna(0)
            continue
        
        for x in (df[column][i]).split(';'):
            entry[x] = np.int64(1)
        
        df_expanded = df_expanded.append(entry, ignore_index=True).fillna(0)
        
    return df_expanded.join(df['ConvertedComp']).set_index(df.index)

# For columns with numerical responses, a data frame is returned with the responses and the salary column.
def num_col(column:str) -> (pd.DataFrame):
    
    for i in range(1,len(df)):
        e = df.loc[i,column]
    
    return df[[column,'ConvertedComp']].astype(np.float64)

# Outliers are removed. An outlier is defined as a response which lies beyound the 3*sigma interval of the mean. This is 
# especially useful for numerical columns. A reject list is created which contains the indices of the responses that should be removed.
def remove_outliers(column:str, reject:list) -> (list):
    
    mean = df.loc[:,column].mean()
    std_3 = 3 * df.loc[:,column].std()
    
    if mean - std_3 < 0:
        minimum = 0.0
    else:
        minimum = mean - std_3
    
    maximum = mean + std_3
    
    for i in range(1,len(df)):
        
        entry = df.loc[i,column]
        
        if column == 'ConvertedComp':
            if (math.isnan(entry) or (entry < minimum) or (entry > maximum)):
                if i not in reject:
                    reject.append(i)
        else:
            if ((entry < minimum) or (entry > maximum)):
                if i not in reject:
                    reject.append(i)
        
    return reject

df = pd.read_csv('../data/survey_results_public.csv', index_col='Respondent')

# Divide columns into multiple, numeric or exclusive
multiple = ['EduOther','DevType','LastInt','JobFactors','WorkChallenge','LanguageWorkedWith','LanguageDesireNextYear',
           'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith','PlatformDesireNextYear',
           'WebFrameWorkedWith','WebFrameDesireNextYear','MiscTechWorkedWith','MiscTechDesireNextYear',
           'DevEnviron','Containers','SOVisitTo','SONewContent','Gender','Sexuality','Ethnicity']

numeric = ['YearsCode','Age1stCode','YearsCodePro','CompTotal','ConvertedComp','WorkWeekHrs','CodeRevHrs','Age']

exclusive = [x for x in df.columns if ((x not in numeric) and (x not in multiple))]

for column in exclusive:
    
    name = 'df_' + column
    path = '../data/columns/exclusive/' + column + '.csv'
    locals()[name] = exc_col(column)
    locals()[name].to_csv(path)

for column in multiple:
    
    name = 'df_' + column
    path = '../data/columns/multiple/' + column + '.csv'
    locals()[name] = mul_col(column)
    locals()[name].to_csv(path)

# Replace some string responses to numerics, to make preprocessing easier
df[['YearsCode','Age1stCode','YearsCodePro']] = df[['YearsCode','Age1stCode','YearsCodePro']].replace(['Less than 1 year','Younger than 5 years',
                                                       'More than 50 years','Older than 85'],
                                                      ['0.5','2.5','56','86']).astype(np.float64)

for column in numeric:
    
    if column != 'ConvertedComp' and column != 'CompTotal':
        name = 'df_' + column
        path = '../data/columns/numeric/' + column + '.csv'
        if column == 'ConvertedComp' or column == 'CompTotal':
            continue
        locals()[name] = num_col(column)
        locals()[name].to_csv(path)

# Get indices to reject
reject = []
for column in numeric:
    
    reject = remove_outliers(column, reject)
    reject.sort()

# Remove rows which were flagged by reject
path = '../data/columns'
for folder in os.listdir(path):
    fold_path = path + '/' + folder
    try:
        for file in os.listdir(fold_path):
            if file.endswith('.csv'):
                file_path = fold_path + '/' + file
                df_temp = pd.read_csv(file_path, index_col='Respondent')
                df_temp = df_temp.drop(index=reject)
                df_temp.to_csv(file_path)     
    except:
        continue

