import pandas as pd
import numpy as np
import math
import os

df = pd.read_csv('../data/survey_results_public.csv', index_col='Respondent')

# Divide columns into three categories, ones which may contain multiple responses in one entry (eg. programming languages used), 
# one for numeric responses (eg. age, salary), and one for exclusive responses (eg. country)
multiple = ['EduOther','DevType','LastInt','JobFactors','WorkChallenge','LanguageWorkedWith','LanguageDesireNextYear',
           'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith','PlatformDesireNextYear',
           'WebFrameWorkedWith','WebFrameDesireNextYear','MiscTechWorkedWith','MiscTechDesireNextYear',
           'DevEnviron','Containers','SOVisitTo','SONewContent','Gender','Sexuality','Ethnicity']

numeric = ['YearsCode','Age1stCode','YearsCodePro','CompTotal','ConvertedComp','WorkWeekHrs','CodeRevHrs','Age']

exclusive = [x for x in df.columns if ((x not in numeric) and (x not in multiple))]

# Function to take exclusive columns and explode them into one-hot-encoded columns, for each response;
# the column for annual salary is also appended to make learning individual features easier
def exc_col(column:str):
    
    df_expanded = (pd.get_dummies(df[column])).join(df['ConvertedComp'])
    return df_expanded

# For multiple-response columns, each entry is taken and split by the delimiter ';', and then exploded into one-hot-encoded
# columns; annual salary column is also appended
def mul_col(column:str):
    
    entry = dict()
    df_expanded = pd.DataFrame()
    for i in list(df.index):
        
        entry = dict()
        if type(df.loc[i,column]) == float and math.isnan(df.loc[i,column]):
            entry[column+'_na'] = np.int64(1)
            df_expanded = df_expanded.append(entry, ignore_index=True).fillna(0)
            continue
        
        for x in (df.loc[i,column]).split(';'):
            entry[x] = np.int64(1)
        
        df_expanded = df_expanded.append(entry, ignore_index=True).fillna(0)
    
    df_expanded = df_expanded.set_index(df.index)
    df_expanded['ConvertedComp'] = df['ConvertedComp']
        
    return df_expanded

# Numerical columns are maintained as is, but for each individual feature, the annual salary column is appended and saved in a separate csv file
def num_col(column:str):
    
    for i in range(1,len(df)):
        e = df.loc[i,column]
    
    return df[[column,'ConvertedComp']].astype(np.float64)

# Function to remove outliers, especially important for numerical columns
# Outliers are defined as values which lie beyond the 3*sigma interval of the mean
# Also rows in which there is no salary data
# The function returns a list of indices which must be discounted while training the NN
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

def main():

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

    df[['YearsCode','Age1stCode','YearsCodePro']] = df[['YearsCode','Age1stCode','YearsCodePro']].replace(['Less than 1 year','Younger than 5 years',
                                                       'More than 50 years','Older than 85'],
                                                      ['0.5','2.5','56','86']).astype(np.float64)

    for column in numeric:
        if column != 'ConvertedComp' and column != 'CompTotal':
            name = 'df_' + column
            path = '../data/columns/numeric/' + column + '.csv'
            locals()[name] = num_col(column)
            locals()[name].to_csv(path)

    reject = []
    for column in numeric:
        reject = remove_outliers(column, reject)
        reject.sort()

    # Save files after removing rejected rows
    path = '../data/columns'
    for folder in os.listdir(path):
        if folder == '.DS_Store':
            continue
        fold_path = path + '/' + folder
        for file in os.listdir(fold_path):
            if file.endswith('.csv') and file != 'ConvertedComp.csv':
                print(file)
                file_path = fold_path + '/' + file
                df_temp = pd.read_csv(file_path, index_col='Respondent')
                try:
                    df_temp = df_temp.drop(index=reject)
                    df_temp.to_csv(file_path)
                except:
                    continue

if __name__== 'main':
    main()