import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Preparing the dataset
dataset = pd.read_csv("Credit_Card_Applications.csv")
X = dataset.iloc[:,-1].values
y = dataset.iloc[:,-1].values

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Creating the SOM using minisom
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)

#Plotting the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's'] 
colors = ['r', 'g'] 
for i,x in enumerate(X):
  w = som.winner(x) 
  plot(w[0]+0.5 , w[1]+0.5 , markers[y[i]], markeredgecolor = colors[y[i]], markerfacecolor = 'None' )
show()
mappings = som.win_map(X) 
fraud = np.concatenate((mappings[(6,6)], mappings[(8,1)]), axis=1)
fraud = sc.inverse_transform(fraud)
print(fraud)

#Building th ANN
customers = dataset.iloc[:,1:].values
is_Fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
  if dataset.iloc[i,0] in fraud:
    is_Fraud[i] = 1

import tensorflow
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'relu'))
ann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'relu'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(customers, is_Fraud, epochs=2, batch_size=1)

y = ann.predict(customers)
y = np.concatenate((dataset.iloc[:,0:1], y), axis=1)