#!/usr/bin/env python
# coding: utf-8

# In[ ]:


__author__ = "Manik Aggarwal"
__email__ = "aggmanik@vt.edu"

#import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgbm
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore")

# function to load the data into a Pandas dataframe 

def read_file(filename):
    dataframe = pd.read_csv(filename)
    print("The shape of dataframe:", dataframe.shape)
    return dataframe

# function to merge dataframes based on column

def joining_tables(df1,df2, columns):
    merged_df = pd.merge(df1,df2, on = [columns])
    return merged_df

## function to remove outliers i.e salary less than zero

def cleaning(df):
    df = df['salary' > 0]
    return df

## Function to Create a new variable for categorising years of experience as follows: 0-2, 2-5, 5-10, 10-15, 15-30

def preprocess_experience(df):
    cut_points = [0, 2, 5, 10, 15, 30]
    label_names = ['0-2','2-5','5-10','10-15','15-30']
    df['years_category'] = pd.cut(df["yearsExperience"],cut_points,labels = label_names, right = False)
    return df

## Function to Create a new variable for categorising miles from metropolis in four categories of size 25.

def preprocess_miles(df):
    cut_points = [0, 25, 50, 75, 100]
    label_names = ['Near','Not Far','Far','Very Far']
    df['miles_category'] = pd.cut(df["milesFromMetropolis"],cut_points,labels = label_names, right = False)
    return df

## Function to calcuate the required metric (mse) for all the models using 5 fold cross validation

def model_metric_scores(estimator,df_X,df_Y,cv_fold,mse_scores):
    estimator_predict = cross_val_predict(estimator,df_X,df_Y, cv=cv_fold)
    mse_scores[estimator] = mean_squared_error(df_Y,estimator_predict)

## function to create dummy variables

def feature_encoding(df, feature_list):
    df_new = pd.get_dummies(df, columns= feature_list, drop_first = True)
    return df_new

## function to drop specific list of columns in the dataframe by passing in the dataframe and column list as arguments

def drop_columns(df, col):
    df = df.drop(col, axis=1)
    return df

## function to create list of numerical and categorical features using dataframe as input argument

def features_set(df):
    numerical_var = df.select_dtypes(include = [np.number]).columns.tolist() ## creating list of numerical features
    categorical_var = df.select_dtypes(include = ['O','category']).columns.tolist()     ## creating list of categorical features
    return numerical_var, categorical_var

## Function to print summary

def print_summary(estimator,mse_scores):
    print('\n Model:\n', estimator)
    print('Mean score of model:\n', mse_scores[estimator])

## Function to save results of model, mse_scores, predictions and feature importances

def save_results(model, mse_scores, test_salaries, feature_importances):
    with open('model.txt', 'w') as file:
        file.write(str(model))
    feature_importances.to_csv('feature_importances.csv')
    test_salaries.to_csv('test_salaries.csv', index = False)
    
    
if __name__ == '__main__':
    
    #define inputs
    
    train_feature_file = 'data/train_features.csv'
    train_target_file = 'data/train_salaries.csv'
    test_feature_file = 'data/test_features.csv'

    #load data
    
    print("Loading data")
    feature_df = read_file(train_feature_file)
    target_df = read_file(train_target_file)
    test_df = read_file(test_feature_file)

    #Joining training data
    print('Consolidating data')
    train_df = joining_tables(feature_df, target_df, key='jobId')

    #clean data by removing outliers from training data and non required columns from training and test data
    print('Cleaning data')
    train_df = cleaning(train_df))
    train_df = drop_columns(train_df,['jobId','companyId'])
    
    test_df_clean = drop_columns(test_df,['jobId','companyId'])
    
    #feature engineering for training and test data
    print('feature engineering')
    train_df = preprocess_experience(train_df)
    train_df = preprocess_miles(train_df)
    
    test_df_clean = preprocess_experience(test_df_clean)
    test_df_clean = preprocess_miles(test_df_clean)
    
    #Create X training and y training dataframe

    y_train = train_df['salary']
    X_train = train_df.drop(['salary'], axis = 1)
       
    #define variables list for training and test data
    
    num_var_train, cat_var_train = features_set(X_train)
    
    num_var_test, cat_var_test = features_set(test_df_clean)

    #encode categorical data and get final feature dfs for training and test dataframe
    print("Encoding data")
    X_train_final = feature_encoding(X_train, cat_var_train)  ### training dataframe
    test_df_final = feature_encoding(test_df_clean, cat_var_test)   ### test dataframe

    #initialize model list and dicts
    models = []
    mean_mse = {}

    #create models -- hyperparameter tuning already done by hand, validation curve and grid search for all models 
    
    lr= make_pipeline(StandardScaler(),LinearRegression())
    ridge = Ridge(random_state= 6, alpha = 100)
    lasso = Lasso(random_state= 7, alpha = 0.01)
    RF = RandomForestRegressor(random_state=27,
                            n_estimators= 40 , max_depth= 20, 
                            min_samples_split = 20, max_features = 0.55,
                            min_samples_leaf = 10, n_jobs = -1)
    
    LGBM = lgbm.LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.1, max_depth=7,
       max_features=1.0, min_child_samples=20, min_child_weight=0.001,
       min_split_gain=0.0, n_estimators=200, n_jobs=-1, num_leaves=70,
       objective='regression', random_state=3, reg_alpha=0.0,
       reg_lambda=0.0, silent=True, subsample=1.0,
       subsample_for_bin=200000, subsample_freq=0)

    models.extend([lr, ridge, lasso, RF, LGBM])

    #Cross validate models using 5 fold cross validation, using MSE as evaluation metric, and print summaries
    print("Cross validation")
    for model in models:
        model_metric_scores(model, X_train_final, y_train, 5, mean_mse)
        print_summary(model, mean_mse)

    #choose model with lowest mse
    model = min(mean_mse, key=mean_mse.get)
    print('\nPredictions calculated using model with lowest MSE:')
    print(model)

    #train model on entire dataset
    
    model.fit(X_train_final, y_train)

    #create predictions based on test data
    predictions = model.predict(test_df_final)
    
    ## create dataframe with predictions and jobId as desired
    test_df['predictions'] = predictions 
    test_salaries = test_df[['jobId','predictions']]

    #store feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        #linear models don't have feature_importances_
        importances = [0]*len(X_train_final.columns)

    feature_importances = pd.DataFrame({'feature':X_train_final.columns, 'importance':importances})
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    
    #set index to 'feature'
    feature_importances.set_index('feature', inplace=True, drop=True)
    
    #create plot
    feature_importances[0:25].plot.bar(figsize=(20,10))
    plt.show()

    #save results
    save_results(model, mean_mse[model], test_salaries, feature_importances)

