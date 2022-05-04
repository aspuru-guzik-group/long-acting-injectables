"""
CODE DEPLOYED IN GOOGLE COLABS
"""

# import the necessary libraries to execute this code
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
import ngboost
from ngboost.distns import Exponential, Normal, LogNormal
from ngboost import NGBRegressor
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

# set the NGBoost regressor to NGB
NGB = NGBRegressor()

# assign some decision tree regressors to be evaluated in the random grid search
b1 = DecisionTreeRegressor(criterion='squared_error', max_depth=2)
b2 = DecisionTreeRegressor(criterion='squared_error', max_depth=4)
b3 = DecisionTreeRegressor(criterion='squared_error', max_depth=8)
b4 = DecisionTreeRegressor(criterion='squared_error', max_depth=12)
b5 = DecisionTreeRegressor(criterion='squared_error', max_depth=16)
b6 = DecisionTreeRegressor(criterion='squared_error', max_depth=32)

# create a dict of paramaters to be serached during random grid search (rnd_search)
param_distribs={
    'n_estimators':[100,200,300,400,500,600,800],
    'learning_rate': [0.1, 0.01, 0.001],
    'minibatch_frac': [1.0, 0.8, 0.5],
    'col_sample': [1, 0.8, 0.5],
    'Base': [b1, b2, b3, b4, b5, b6]
}

# run rnd_search applying logo for each of the 100 interation and scoring based on mean absolute error
rnd_search_cv = RandomizedSearchCV(NGB, param_distribs, n_iter=100,verbose = 3, cv=logo, scoring='neg_mean_absolute_error')
rnd_search_cv.fit(X=X, y=Y, groups=G)

# return best paramater for rnd_search
rnd_search_cv.best_params_

# return best score for rnd_search
rnd_search_cv.best_score_

# train NGB based on best paramaters identified during rnd_search and return model structure
regressor=rnd_search_cv.best_estimator_
regressor.fit(X=X,Y=Y)

# saved trained NGB model to pickle file

Pkl_Filename = "NGB.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(regressor, file)

# Load trained model and return structre
with open("NGB.pkl", 'rb') as file:  
            NGB_Model = pickle.load(file)

NGB_Model

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

#convert to np array
Valid_X=np.array(Valid_X)

# apply the trained model to predict fractional drug release for every entry in the external validation dataset, and return the predictions as a list
Y_preds = NGB_Model.predict(Valid_X)
# apply the trained model to predict the confidence for every prediction and return the predictions as a list
Y_dists = NGB_Model.pred_dist(Valid_X)

# returns a dict of mean predictions and standard deviation
df_results = Y_dists.params
# returns an array of mean predictions
df_results['loc']
# returns an array of standard deviation
df_results['scale']

# Get list of tuples from lists: DPCombo | Experimental | Predicted | Confidence
data_tuples = list(zip(G,Y, df_results['loc'],df_results['scale']))
# Create a dataframe from the list of tuples
target_result = pd.DataFrame(data_tuples, columns=['DPCombo','Experimental','Predicted','STDEV'])
# Calculate mean absolute error (MAE) and save as column in new dataframe
target_result['MAE'] = (target_result['Predicted'] - target_result['Experimental']).abs()

# display first df
target_result

# save this new dataframe to an excel file
target_result.to_excel(r'Path where the Excel file will be saved\File name.xlsx', index = False)

# define a function to return the mean absolute error between two lists
def Scores(y_observed,y_pred):  
    print('mean_absolute_error:            ',mean_absolute_error(y_observed, y_pred))
    return

# define this fucntion to the predicted and experimental fractional drug release values
Scores(target_result['Experimental'],target_result['Predicted'])