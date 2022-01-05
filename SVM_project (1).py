# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 22:47:04 2019

@author: Mypc
"""

import pandas as pd 
import numpy as np
import math
import statistics
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import Imputer
imputer= Imputer(missing_values="NaN", strategy='mean', axis=0)
missing_value = '?'
dataset = pd.read_csv('breast-cancer-wisconsin.data.csv', na_values = missing_value)
X = dataset.iloc[:,1:].values
imputer.fit(X)
X=imputer.transform(X)
dfX = pd.DataFrame(X)

x = dataset.iloc[:, 1:10].values
imputer.fit(x)
x=imputer.transform(x)
y = dataset.iloc[:, 10].values

dfx = pd.DataFrame(x)
c=0

'''for i in range (len(x)):
    for j in range (len(x[0])):
        if x[i][j]=='?6':
            x[i][j] =-1
        else:
            x[i][j] = float(x[i][j])'''



for i in range (len(y)):
    if y[i] == 4:
        y[i]= 1
    elif y[i] == 2:
        y[i] = -1
    
omega = []
for i in range(9):
    omega.append(random.randint(100,200))

def kernel(a1,b1,og):
    sum=0
    for i in range (len(a1)):
        sum+=(float(a1[i])-float(b1[i]))**2*og[i]
    value = math.exp(-sum)
    return value
f= kernel(x[1],x[2],omega)

def fx (alpha,y,x,xj,b):
    sum= 0
    for i in range (len(y)):
        sum+= alpha[i]*y[i]*kernel(x[i],xj,omega)
    value = sum + b
    return value

def error (f , b3):
    E = f - b3
    return E

def eta (a,b):
    n = 2* kernel(a,b,omega) - kernel(a,a,omega) - kernel(b,b,omega)
    return n

def solve_alpha(x,y):
    control = 0.5
    tol = 0.05
    max_passes = 50
    
    alpha = [0 for i in range(len(x))]
    b=0
    passes = 0
    
    while (passes< max_passes):
        num_changed_alphas=0
        for i in range (len(x)):
            Ei = fx(alpha,y,x,x[i],b)-y[i]
            if (y[i]*Ei< -tol and alpha[i]<control) or (y[i]*Ei> tol and alpha[i]>0):
                j = random.randint(0,len(x)-1)
                while(j==i):
                    j = random.randint(0, len(x)-1)
                Ej = fx(alpha,y,x,x[j],b)-y[j]
                alpha_old_i = alpha[i]
                alpha_old_j = alpha[j]
                if (y[i]!=y[j]):
                    L = max(0,alpha[j]-alpha[i])
                    H = min(control, control+ alpha[j]-alpha[i])
                else :
                    L = max(0, alpha[i]+alpha[j]-control)
                    H = min(control, alpha[i]+alpha[j])
                if (L == H):
                    continue
                n = eta(x[i],x[j])
                if (n >= 0):
                    continue
                alpha[j] = alpha[j] - y[j]*(Ei - Ej)/n
                if (alpha[j]>H):
                    alpha[j] = H
                elif (alpha[j]< L):
                    alpha[j] = L
                if (abs(alpha[j]-alpha_old_j)< 10**-5):
                    continue
                alpha[i] = alpha[i]+y[i]*y[j]*(alpha_old_j- alpha[j])
                b1 = b-Ei-y[i]*(alpha[i]-alpha_old_i)*kernel(x[i],x[i],omega)-y[j]*(alpha[j]-alpha_old_j)*kernel(x[i],x[j],omega)
                b2 = b-Ej-y[i]*(alpha[i]-alpha_old_i)*kernel(x[i],x[j],omega)-y[j]*(alpha[j]-alpha_old_j)*kernel(x[j],x[j], omega)
                if (alpha[i]<control and alpha[i]>0):
                    b = b1
                elif (alpha[j]<control and alpha[j]>0):
                    b = b2
                else:
                    b = (b1+b2)/2
                num_changed_alphas = num_changed_alphas+1
        if(num_changed_alphas==0):
            passes = passes+1
        else:
            passes = 0

    return alpha

#solve for y_hat score
def sol_func(alpha,x,y,xi):
    sum=0
    for i in range(len(x)):
        sum+=alpha[i]*y[i]*kernel(xi,x[i],omega)
    return sum

def pearson_corr(x,y):
    sum_x = 0
    sum_y = 0
    sum_xy = 0
    sqrsum_x = 0
    sqrsum_y = 0
    n = len(x)
    for i in range(n):
        sum_x += x[i]
        sum_y += y[i]
        sum_xy += x[i]*y[i]
        sqrsum_x += x[i]*x[i]
        sqrsum_y += y[i]*y[i]
    
    corr = (n*sum_xy-sum_x*sum_y)/math.sqrt((n*sqrsum_x-sum_x*sum_x)*(n*sqrsum_y-sum_y*sum_y))
    return corr

def solve_theta(x,y,alpha,theta):
    scores = []
    for i in range(len(x)):
        val = sol_func(alpha,x,y,x[i])
        scores.append(val)
    corr = pearson_corr(y,scores)
   


'''model = KMeans(n_clusters = 10, init = 'k-means++', n_init = 10, max_iter = 300, random_state = 0)
z = model.fit_predict(x)

dict = [[]]

x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
x9 = []

for i in range(len(x)):
    if(z[i]==0):
        x0.append(x[i])
    if(z[i]==1):
        x1.append(x[i])
    if(z[i]==2):
        x2.append(x[i])
    if(z[i]==3):
        x3.append(x[i])
    if(z[i]==4):
        x4.append(x[i])
    if(z[i]==5):
        x5.append(x[i])
    if(z[i]==6):
        x6.append(x[i])
    if(z[i]==7):
        x7.append(x[i])
    if(z[i]==8):
        x8.append(x[i])
    if(z[i]==9):
        x9.append(x[i])
'''
l1 = int(len(x)/4)
l2 = int(len(x)/2)
l3 = int(3*len(x)/4)
s1 = []
s2 = []
s3 = []
s4 = []
sy1 = []
sy2 = []
sy3 = []
sy4 = []

for i in range(l1):
    s1.append(x[i])
    sy1.append(y[i])

for i in range(l1,l2):
    s2.append(x[i])
    sy2.append(y[i])

for i in range(l2,l3):
    s3.append(x[i])
    sy3.append(y[i])

for i in range(l3,len(x)):
    s4.append(x[i])
    sy4.append(y[i])
    
#Hill climbing algorithm
def hill_climb(w_omega,x,y):
   
C = [0.5,0.6,0.7]
H = 3
Phi =  [[2],[2,3,25],[2,3,4,27,90]]
acc = 0
for c in C:
    h=1
    theta=Phi[random.randint(0,len(Phi)-1)]
    while h<H:
        #for s1
        alpha = solve_alpha(s1,sy1)
        theta_opt = solve_theta(s1,sy1,alpha,theta)
        l = random.randint(1,h)
        i=0
        while i<l:
            i+=1
        
        theta_opt.insert(i , theta_opt[i])
        
        h+=1
        #l=random.randint(0,h)
Alpha = []
new_omega = 0
Alpha.append(alpha)
scores_pre = []
for i in range (173):
    scores_pre.append(sol_func(alpha,s1,y,x[i]))
coefficent = pearson_corr(scores_pre, sy1)
new_omega = hill_climb(omega,s1,sy1)
        