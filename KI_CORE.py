# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:52:08 2019

@author: obr_mohammade
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydot
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adam,SGD
from keras.utils.vis_utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier
from keras import metrics



# loading the data
df = pd.read_excel("loan_data_final.xlsx")
df.index = df["InstrumentID"]
df = df.drop(["InstrumentID","TradingAreaDesc"],axis = 1)
df.head()
df.shape
df.dtypes
df.describe()
df.corr()
df.skew()


# creating dummy variables
df.dtypes
cat_features = ["SectorOfCounterparty","Typ","LegType","AccountType","IFRS9","BucketALMM1",
                "BucketNSFR","BucketStoraExponeringar","PrincipalAmortType","CRD1Class2",
                "BucketRemainingMaturity","RiskWeightBucket"]
df_final = pd.get_dummies(data = df,columns =cat_features,drop_first = True).drop(['BOLink', 'BONote1', 'MunicipalityCode'],axis = 1)

# splitting the dataset
x = df_final.drop("has_swap_deal",axis = 1)
y = df_final["has_swap_deal"]
test_size = 0.2
seed = 2019
x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                 test_size = test_size,
                                                 random_state = seed)

# scalling the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
x_train_scaled = pd.DataFrame(data = x_train_scaled,columns = x_train.columns)
x_test_scaled = pd.DataFrame(data = x_test_scaled, columns = x_test.columns)


# kfold cross validation with gridsearch
# 1. parameter grid
units = [40,44,50,60]
dropout_rate = [0.1,0.2,0.3]
lr = [0.0001,0.001,0.01,0.1]
decay =[0.0,0.01]

# 2 hyperparameter grid
batch_size = [25,32]
epochs = [100,500]


param_grid = dict(batch_size = batch_size,
                  epochs = epochs,
                  dropout_rate = dropout_rate,
                  lr = lr,
                  decay =decay,
                  units = units)

def create_classifier(units,dropout_rate,lr,decay):
    ann_classifier = Sequential()
    ann_classifier.add(Dense(                 # 1st hidden layer
            input_dim = 87,
            units= units, 
            activation = "relu",
            kernel_initializer = "uniform"
            ))
    ann_classifier.add(Dropout(dropout_rate))
    ann_classifier.add(Dense(                 # 2nd hidden layer
            units = units,
            activation = "relu",
            kernel_initializer = "uniform"
            ))
    ann_classifier.add(Dropout(dropout_rate))
    ann_classifier.add(Dense(
            units = 1,
            kernel_initializer= "uniform",   # output layer
            activation = "sigmoid"
            ))
    optimizer = Adam(lr,decay)
    ann_classifier.compile(
            optimizer = optimizer,
            loss = "binary_crossentropy",
            metrics = ["acc"]
            )
    return ann_classifier

ann_classifier = KerasClassifier(build_fn = create_classifier)

# visualization of the model
print(ann_classifier.summary())
plot_model(ann_classifier, to_file='ann_classifier_plot.png', show_shapes=True, show_layer_names=True)


kfold = KFold(n_splits = 10, shuffle = True, random_state = seed)
grid = GridSearchCV(estimator = ann_classifier,
                    param_grid = param_grid,
                    scoring = "accuracy",
                    cv = kfold,
                    verbose = 1
                    )

grid_results = grid.fit(X = x_train_scaled, y = y_train)
ann_classifier.save("ann_adam.h5")
