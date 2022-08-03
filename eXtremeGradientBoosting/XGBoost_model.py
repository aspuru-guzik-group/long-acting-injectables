"""
CODE DEPLOYED IN JUPYTER NOTEBOOK
"""

# import the necessary libraries to execute this code
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut,GroupKFold
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import pickle

# import training dataset from excel file as a pandas dataframe 
dataset = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")

# split training data into features(X), targets(Y), and drug-polymer groups(G)
X = dataset.drop(['DP_Group','Release'],axis='columns')
Y = dataset['Release']
G = dataset['DP_Group']

# apply standard scaler to features
stdScale = StandardScaler().fit(X)
X=stdScale.transform(X)

# split data according into train/test indices by drug-polymer group.
logo = LeaveOneGroupOut()

for train_index, test_index in logo.split(X, Y, G): 
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

# set Extreme Gradient Boosting regressor to XGB
XGB = XGBRegressor()

# create a dict of paramaters to be serached during random grid search (rnd_search)
param_distribs={
    'booster': ['gbtree', 'gblinear', 'dart'],
    "n_estimators":[100, 150, 300, 400, 500, 600],
    'max_depth':[6, 15, 30, 40, 60],
    'gamma':[0, 2, 4, 6, 8, 10],
    'learning_rate':[0.1, 0.01, 0.001, 0.0001],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1.0, 2.0, 4.0, 5.0, 10.0],
    'max_delta_step':[1, 2, 4, 6, 8, 10],
    'reg_alpha':[0.001, 0.01, 0.1, 0.5],
    'reg_lambda': [0.001, 0.01, 0.1, 0.5]
}

# run rnd_search applying logo for each of the 250 interation and scoring based on mean absolute error
rnd_search_cv = RandomizedSearchCV(XGB, param_distribs, n_iter=250,verbose = 3, cv=logo, scoring='neg_mean_absolute_error')
rnd_search_cv.fit(X=X, y=Y, groups=G)

# return best paramater for rnd_search
rnd_search_cv.best_params_

# return best score for rnd_search
rnd_search_cv.best_score_

# train XGBoost based on best paramaters identified during rnd_search and return model structure
regressor=rnd_search_cv.best_estimator_
regressor.fit(X=X,y=Y)

# save trained XGBoost model to pickle file
Pkl_Filename = "XGBoost.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(regressor, file)

# Load trained model and return structre
with open("XGBoost.pkl", 'rb') as file:  
            model = pickle.load(file)

# import external validation dataset from excel file as a pandas dataframe 
dataset = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")

# create new dataframe by droping the column experimental index from the external validation dataset
dataset_2 = dataset.drop(['Experiment_index'], axis='columns')

# split the external validation dataset into features(Valid_X), targtes(Y), and drug-polymer groups(G)
Valid_X = dataset_2.drop(['DP_Group','Release'],axis='columns')
Y = dataset_2['Release']
G = dataset_2['DP_Group']

# apply standard scaler to features and convert to np array
stdScale = StandardScaler().fit(Valid_X)
Valid_X=stdScale.transform(Valid_X)
Valid_X=np.array(Valid_X)

# apply the trained model to predict fractional drug release for every entry in the external validation dataset, and return the predictions as a list
result = model.predict(Valid_X)
result = result.flatten()

# merge three lists together as a list of tuples: DPCombo | Experimental | Predicted
data_tuples = list(zip(G,Y,result))
# create a dataframe from the list of tuples
target_result = pd.DataFrame(data_tuples, columns=['DPCombo','Experimental','Predicted'])
# calculate mean absolute error (MAE) and save as column in new dataframe
target_result['MAE'] = (target_result['Predicted'] - target_result['Experimental']).abs()
# display this dataframe
target_result

# save this new dataframe to an excel file
target_result.to_excel(r'Path where the Excel file will be saved\File name.xlsx', index = False)

# define a function to return the mean absolute error between two lists
def Scores(y_observed,y_pred):  
    print('mean_absolute_error:            ',mean_absolute_error(y_observed, y_pred))
    return
# define this fucntion to the predicted and experimental fractional drug release values
Scores(target_result['Experimental'],target_result['Predicted'])
