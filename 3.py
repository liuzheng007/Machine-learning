# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:41:51 2017

@author: Boardwell
"""

import numpy as np
#import matplotlib.pyplot as plt
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
l=10#output dimension
q=100#number of hidden layer
yita=0.1
v=np.random.rand(d,q)
w=np.random.rand(q,l)
v=v*0.1-0.05
v=np.matrix(v)
w=w*0.1-0.05
w=np.matrix(w)
alpha=np.array([0.0]*q)
beta=np.array([0.0]*l)
b=np.array([0.0]*q)
b=np.matrix(b)
theta=np.random.rand(l)
theta=np.matrix(theta)
gamma=np.random.rand(q)
gamma=np.matrix(gamma)
g=np.array([0.0]*l)
e=np.array([0.0]*q)
accuracy_now=0.0
accuracy=[]
def sigmoid(t):
    re=1/(1+np.exp(-t))
    return re
#fitting the parameters
for k in range(30):
    if k>=10:
        yita=0.05
    if k>=20:
        yita=0.02
    for s in range(n):
        x=data[s]
        x=np.matrix(x)
        y_targets=np.zeros(l)
        y_predict=np.zeros(l)
        y_targets[targets[s]]=1
        alpha=x*v
        temp_b=np.array(b)
        temp_alpha=np.array(alpha)
        temp_gamma=np.array(gamma)
        for h in range(q):
            temp_b[0][h]=sigmoid(temp_alpha[0][h]-temp_gamma[0][h])
        b=np.matrix(temp_b)
        beta=np.array([0.0]*l)
        beta=np.matrix(beta)
        beta=b*w
        temp_beta=np.array(beta)
        temp_theta=np.array(theta)
        for j in range(l):
            y_predict[j]=sigmoid(temp_beta[0][j]-temp_theta[0][j])
        g=y_predict*(1-y_predict)*(y_targets-y_predict)
        g=np.matrix(g)
        e=np.array([0.0]*q)
        e=np.matrix(e)
        e=g*w.T
        temp_b=np.array(b)
        temp_e=np.array(e)
        for h in range(q):
            temp_e[0][h]=temp_b[0][h]*(1-temp_b[0][h])*temp_e[0][h]
        e=np.matrix(temp_e)
        theta=theta-yita*g
        w=w+yita*b.T*g
        gamma=gamma-yita*e
        v=v+yita*x.T*e
#test the result    
test=np.genfromtxt('test_data.csv',delimiter = ',',dtype=float)
testshape=test.shape
result=np.array([[0]*1]*testshape[0]) 
for s in range(testshape[0]):
    x=test[s]
    x=np.matrix(x)
    y_targets=np.zeros(l)
    y_predict=np.zeros(l)
    y_targets[targets[s]]=1
    alpha=x*v
    temp_b=np.array(b)
    temp_alpha=np.array(alpha)
    temp_gamma=np.array(gamma)
    for h in range(q):
        temp_b[0][h]=sigmoid(temp_alpha[0][h]-temp_gamma[0][h])
    b=np.matrix(temp_b)
    beta=np.array([0.0]*l)
    beta=np.matrix(beta)
    beta=b*w
    temp_beta=np.array(beta)
    temp_theta=np.array(theta)
    for j in range(l):
        y_predict[j]=sigmoid(temp_beta[0][j]-temp_theta[0][j])
    max=0
    for j in range(l):
        if y_predict[j]>y_predict[max]:
            max=j
    result[s][0]=max
#write to file       
with open('test_predictions.csv','w',newline='') as csvfile:
        spamwriter= csv.writer(csvfile,dialect='excel')
        spamwriter.writerows(result)
'''
time_end=time.time()
print(time_end-time_start,"s")
'''