"""
CODE DEPLOYED IN JUPYTER NOTEBOOK
"""

# import the necessary libraries to execute this code
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
import forestci as fci
import pickle

# import training dataset from excel file as a pandas dataframe for correlation analysis
training_data = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")
# create dataframe "X" with input features only
X = training_data.drop(['DP_Group','Release'],axis='columns')
# create dataframe "Y" with training target values only
Y = training_data['Release']
# create dataframe "G" with drug-polymer groups from the training dataset
G = training_data['DP_Group']

# preproccessing of training data inputs with standard scalar
stdScale = StandardScaler().fit(X)
X=stdScale.transform(X)
#convert to np array
X=np.array(X)

# import the validation dataset
dataset_valid = pd.read_excel(r"Path where the Excel file is stored\File name.xlsx")
# create dataframe "Valid_X" with input features only
Valid_X_features = dataset_valid.drop(['Experiment_index','DP_Group','Release'], axis='columns')
Valid_X = dataset_valid.drop(['Experiment_index','DP_Group','Release'], axis='columns')
# create dataframe "Y_true" with validation target values only
Y_true = dataset_valid['Release']
# create dataframe "G2" with drug-polymer groups from the validation dataset
G2 = dataset_valid ['DP_Group']

# preproccessing of validation data inputs with standard scalar
stdScale = StandardScaler().fit(Valid_X)
Valid_X=stdScale.transform(Valid_X)
#convert to np array
Valid_X=np.array(Valid_X)

# load RF model hyperparamters from pickle file and return structre
with open(r"Path where the pickle file is stored\File name.pkl", 'rb') as file:  
            RF_model = pickle.load(file)
RF_model

# generate a correlation matrix
corr = spearmanr(X).correlation
# ensure the correlation matrix is symmetric
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
# convert the correlation matrix to a distance matrix 
distance_matrix = 1 - np.abs(corr)
# generate Ward's linkage values for hierarchical clustering
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# empty list to store MAE values
MAE_list = []
# empty list to store features names
feature_name_list = []
# empty list to store number of features
feature_number_list = []
# empty list to store the Ward'slinkage distance
linkage_distance_list = []

# for loop that interates evaluates different RF model iterations based on their Ward linkages 
# and appends the resulting MAE values, number of features, feature names, and linkage distances
# to a series of lists
for n in range(0, 10, 1):
    
    # select input features to be included in this model iteration based on Ward's linkage of n/10
    cluster_ids = hierarchy.fcluster(dist_linkage, (n/10), criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list) 
    for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

    # select input features from original training and validation datasets based on Ward's Linkage value
    X_train_sel = X[:, selected_features]
    X_test_sel = Valid_X[:, selected_features]
    
    # train RF model, with fixed hyperparamters, based on selected input features
    clf_sel = RF_model
    clf_sel.fit(X_train_sel, Y)
    # predict validation dataset based on selected input features
    y_pred = clf_sel.predict(X_test_sel)

    # append linkage distance to empty list and print value
    linkage_distance_list.append(n/10)
    print(linkage_distance_list)
    
    # append MAE value to empty list and print value
    MAE_list.append(round(mean_absolute_error(y_pred,Y_true), 3))
    print(MAE_list)
    
    # create empty list to save feature names
    tested_features = []
    # for loop to append the utilized input feature names to the empty list
    for feature in selected_features:
        tested_features.append(Valid_X_features.columns[feature])
    
    # append the number of input features to empty list and print value
    feature_number_list.append(len(tested_features))
    print(feature_number_list)
    
    # append the list of feature names to an empty list of lists and print value
    feature_name_list.append(tested_features)
    print(feature_name_list)

# create a list of tuples linkage distance, MAE value, feature number and feature names
data_tuples = list(zip(linkage_distance_list, MAE_list, feature_number_list, feature_name_list))
# create a dataframe from the list of tuples with the column names below
model_result = pd.DataFrame(data_tuples, columns=["Ward's linkage values",'MAE','Number of Features','Features Names'])
# save the new dataframe as an excel file
model_result.to_excel(r"Path where the Excel file is to be saved\File name.xlsx", index = False)

# remove model iterations with MAE greater than the 14-feature model
df_new = model_result[(model_result['MAE'] <= model_result.iloc[0,1])]
# select model iteration with the lowest number of input features
df_new = df_new[(df_new['Number of Features'] < df_new.iloc[0,2])]
# return new dataframe
df_new

# only select input features from the 12-feature model from training dataframe
X = training_data[feature_name_list[1]]
# set dataframe "Y" as the training target values
Y = training_data['Release']
# create dataframe "G" with drug-polymer groups from the training dataset
G = training_data['DP_Group']

# preproccessing of training data inputs with standard scalar
stdScale = StandardScaler().fit(X)
X=stdScale.transform(X)
#convert to np array
X=np.array(X)

# only select input features from the 12-feature model from validation dataframe
Valid_X = dataset_valid[feature_name_list[1]]
# create dataframe "Y_true" with validation target values only
Y_true = dataset_valid['Release']
# create dataframe "G2" with drug-polymer groups from the validation dataset
G2 = dataset_valid ['DP_Group']

# preproccessing of validation data inputs with standard scalar
stdScale = StandardScaler().fit(Valid_X)
Valid_X=stdScale.transform(Valid_X)
#convert to np array
Valid_X=np.array(Valid_X)

# train the 12-feature RF model with the saved hyperparamaters
XII_feature_RF = RF_model.fit(X, Y)

# predict drug release for the 12-feature RF model
y_preds = XII_feature_RF.predict(Valid_X)
y_preds = y_preds.flatten()
# calculate the prediction variance for the 12 feature RF model
variance = fci.random_forest_error(XII_feature_RF, X, Valid_X)

# create a list of tuples for drug-polymer groups, obsevered release values, predicted release values, and variance values
data_tuples = list(zip(G2, Y_true, y_preds, variance))
# create a dataframe from the list of tuples with the column names below
target_result = pd.DataFrame(data_tuples, columns=['DPCombo','Experimental','Predicted', 'Variance'])
# calculate mean absolute error (MAE) and save as column in new dataframe
target_result['MAE'] = (target_result['Predicted'] - target_result['Experimental']).abs()
# add experimental index to the new dataframe
target_result['Experiment_index'] = (dataset_valid['Experiment_index'])
# add time values to the new dataframe
target_result['Time'] = (dataset_valid['Time'])
# save the new dataframe as an excel file
target_result.to_excel(r"Path where the Excel file is to be saved\File name.xlsx", index = False)