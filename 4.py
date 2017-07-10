# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:29:54 2017

@author: Boardwell
"""
import numpy as np
def predict(Xt,model):
    result=[]
    a=model.dual_coef_
    b=model.intercept_ 
    for i in range(Xt.shape[0]):
        f=0
        support=model.support_vectors_
        for k in range(a.shape[1]):
            s=0
            for j in range(Xt.shape[1]):
                s+=(support[k][j]-Xt[i][j])**2
            kernel=np.exp(-model.gamma*s)
            f+=kernel*a[0][k]
        f+=b
        if (f>0):
            result.append(1.0) 
        else:
            result.append(0.0) 
        
    return result