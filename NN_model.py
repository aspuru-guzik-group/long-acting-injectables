"""
CODE DEPLOYED IN GOOGLE COLABS
"""

# import the necessary libraries to execute this code
import numpy as np
import pandas as pd
import tensorflow
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Dense,Dropout
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

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

# create a function that will build and compile a Keras model
def NN_builder(n_hidden=1, optimizer = 'rmsprop', units=40, learning_rate = 0.001, input_shape=[14], 
               regularization=0.001, dropout=0.2, activation = 'sigmoid'):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
  
    for layer in range (n_hidden):
        model.add(keras.layers.Dense(units=40, activation=activation, activity_regularizer=l1_l2(regularization)))
        model.add(Dropout(dropout))
  
    model.add(keras.layers.Dense(units=1,activation='sigmoid'))
    optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(loss="mean_absolute_error", optimizer=optimizer)
    return model

# set neural network regressor to NN
NN = tensorflow.keras.wrappers.scikit_learn.KerasRegressor(NN_builder)

# create a dict of paramaters to be serached during random grid search (rnd_search)
param_distribs = {
    "n_hidden" : [1,2,3],
    "units" : [10,20,30],   
    "learning_rate": [0.001,0.01,0.1],
    "regularization":[1e-2,1e-3,1e-4],
    "dropout":[0.0,0.1,0.2],
    "batch_size":[10,20,40],
    "activation": ['softmax', 'relu', 'tanh', 'sigmoid']
}

# run rnd_search applying logo for each of the 50 interation and scoring based on mean absolute error
epochs = 150
callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
rnd_search_cv = RandomizedSearchCV(NN, param_distribs, n_iter=50, cv=logo)
rnd_search_cv.fit(np.asarray(X), np.asarray(Y), groups=G, callbacks=callbacks,epochs=epochs)

# return best paramater for rnd_search
rnd_search_cv.best_params_

# return best score for rnd_search
rnd_search_cv.best_score_

# train NN based on best paramaters identified during rnd_search and return model structure
callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=5)]
regressor=rnd_search_cv.best_estimator_
regressor.fit(np.asarray(X), np.asarray(Y), callbacks=callbacks, epochs=epochs)

# save trained NN model
regressor.model.save("NN_logo.h5")

# Load trained model and return structre
model = keras.models.load_model('NN_logo.h5')

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