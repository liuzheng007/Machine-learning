# -*- coding: utf-8 -*-
"""
Created on Sun May 28 12:48:45 2017

@author: Boardwell
"""
import numpy as np
import pickle
import csv
X=pickle.load(open('train_data.pkl','rb')).todense() # unsupported in Python 2
y=pickle.load(open('train_targets.pkl','rb'))
Xt=pickle.load(open('test_data.pkl','rb')).todense()
X=np.array(X)
Xt=np.array(Xt)
p_y=np.array([1.0]*5)
pxy=np.array([[[1.0]*2]*2500]*5)
mu=np.array([[0.0]*2500]*5)
sigma=np.array([[0.0]*2500]*5)
consist=np.array([[[1.0]*2]*2500]*5)
for i in range(y.shape[0]):
    p_y[y[i]]+=1
    for j in range(2500):
        pxy[y[i]][j][int(X[i][j])]+=1   
    for j in range(2500,5000):
        mu[y[i]][j-2500]+=X[i][j]

for i in range(5):
    pxy[i]=np.log(pxy[i])-np.log(p_y[i]+2)
    mu[i]=mu[i]/(p_y[i])
    
for i in range(y.shape[0]):    
    for j in range(2500,5000):
        sigma[y[i]][j-2500]+=(X[i][j]-mu[y[i]][j-2500])**2

for i in range(5):
    sigma[i]=sigma[i]*1/(p_y[i]-1)    
sigma=np.sqrt(sigma)
for i in range(5):
    for j in range(2500):
        if sigma[i][j]==0:
            sigma[i][j]=0.01

p_y=p_y/float(y.shape[0]+5)

result=np.array([[0]*1]*Xt.shape[0])
for i in range(Xt.shape[0]):
    temp=np.array([0.0]*5)
    maximum=0
    for k in range(5):
        temp[k]+=np.log(p_y[k])
        for j in range(2500):
            temp[k]+=pxy[k][j][int(Xt[i][j])]
            temp[k]+=-np.log(np.sqrt(2*np.pi)*sigma[k][j])-(Xt[i][j+2500]-mu[k][j])**2/(2*sigma[k][j]**2)
        if temp[maximum]<temp[k]:
            maximum=k
    result[i][0]=maximum

with open('test_predictions.csv','w',newline='') as csvfile:
        spamwriter= csv.writer(csvfile,dialect='excel')
        spamwriter.writerows(result)


