# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:58:47 2017

@author: Boardwell
"""

import numpy as np
import csv
from sklearn.model_selection import KFold
#import data
origin_data = np.genfromtxt('data.csv',delimiter = ',',dtype=float)
a=origin_data.shape
origin_target=np.genfromtxt('targets.csv',delimiter = ',',dtype=float)
kf=KFold(n_splits=10, shuffle=False, random_state=None)
times=1
for train, test in kf.split(origin_data):
    b=train.shape
    data=np.array([[1.0]*(a[1]+1)]*b[0])
    targets=np.array([0.0]*b[0])
    for i in range(b[0]):
        targets[i]=origin_target[train[i]]
        for j in range(a[1]):
            data[i][j]=origin_data[train[i]][j]
    

    '''        
    for j in range(a[1]):
        max=0.0
        for i in range(b[0]):
            if max<np.abs(data[i][j]):
                max=np.abs(data[i][j])
        for i in range(b[0]):
            data[i][j]=data[i][j]/max
    '''           
    #Logistic Regression 
    m=a[0]
    n=a[1]
    beta=np.array([0.0]*(n+1))
    beta=np.matrix(beta)
    beta=beta.T
    x=np.array([0.0]*(n+1))
    for i in range(b[0]):
        data[i][n]=1
    L1=np.matrix([[0.0]*1]*(n+1))
    L2=np.matrix([[0.0]*(n+1)]*(n+1))
    p=0.0
    time=0
    for k in range(5):    
        #caculate First derivative of L function
        sum1=np.array([[0.0]*1]*(n+1))
        sum1=np.matrix(sum1)
        for i in range(b[0]):
            s=0.0
            y=np.matrix(data[i])
            y=y.T
            s=beta.T*y
            temp=float(s[0][0])
            p1=np.exp(temp)/(np.exp(temp)+1)
            sum1=sum1+y*(targets[i]-p1)
        L1=np.matrix(-sum1)
        #caculate second derivative of L function
        sum2=np.zeros((n+1,n+1))
        sum2=np.matrix(sum2)
        for i in range(b[0]):
            s=0.0
            x=np.matrix(data[i])
            x=x.T
            s=beta.T*x
            temp=float(s[0][0])
            p1=np.exp(temp)/(np.exp(temp)+1)
            sum2=sum2+x*x.T*p1*(1-p1)
        L2=sum2
        beta_last=beta
        beta=beta-np.linalg.inv(L2)*L1

    c=test.shape
    result=np.array([[0.0]*2]*c[0])
    for i in range(c[0]):
        result[i][0]=test[i]+1
    test_set=np.array([[0.0]*(a[1]+1)]*c[0])
    for i in range(c[0]):
        for j in range(a[1]):
            test_set[i][j]=origin_data[test[i]][j]
    for i in range(c[0]):
        test_set[i][n]=1
    for i in range(c[0]):
        x=np.matrix(test_set[i])
        x=x.T
        s=beta.T*x
        if s<0:
            result[i][1]=0
        else:
            result[i][1]=1
    with open('fold%d.csv'%times,'w',newline='') as csvfile:
        spamwriter= csv.writer(csvfile,dialect='excel')
        spamwriter.writerows(result)
    times=times+1
    
    
    
