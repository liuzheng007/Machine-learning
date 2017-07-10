# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 09:22:43 2017

@author: Boardwell
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import csv
'''
import time 
time_start=time.time()
'''
#import data
data = np.genfromtxt('train_data.csv',delimiter = ',',dtype=float)
targets=np.genfromtxt('train_targets.csv',delimiter = ',',dtype=int)
a=data.shape

n=a[0]#number of test data
d=a[1]#input dimension
y_targets=np.zeros((n,10))
for s in range(n):    
    y_targets[s,targets[s]]=1
#set the model and train the parameters
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=d))
model.add(Dense(512, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, y_targets, epochs=80, batch_size=30,verbose=0)
#test the result
test=np.genfromtxt('test_data.csv',delimiter = ',',dtype=float)
b=test.shape[0]
prediction=model.predict(test)
result=np.array([[0]*1]*b)
for s in range(b): 
    max=0
    for j in range(10):
        if prediction[s][j]>prediction[s][max]:
           max=j
    result[s][0]=max
#write to file
with open('test_predictions_library.csv','w',newline='') as csvfile:
        spamwriter= csv.writer(csvfile,dialect='excel')
        spamwriter.writerows(result)
'''
time_end=time.time()
print(time_end-time_start,"s")
'''