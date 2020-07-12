#%% import
import numpy as np
import pandas as pd 
from numpy import dot

#%% Gradient Descendent Linear Regression
'''
Reference: https://blog.csdn.net/m0_38075425/article/details/90767163

theta_{i+1} = theta_{i} + delta(J(theta)) / delta(theta) * alpha 

Vector of coefficients : theta
Loss function : J(theta) = 1/2m \sum_{i=0}^{m} (h(x_i) - y_i)^2
Gradient : delta(J(theta)) / delta(theta) = 1/m \sum_{i=0}^{m} (h(x_i)-y_i)*x_i

'''
def LR_gradient(X,Y,alpha=0.1,num_iter=100):
    X = np.array(X).reshape(-1,1)
    Y = np.array(Y).reshape(-1,1)
    sample_num = X.shape[0]
    coef_num = X.shape[1]

    theta = np.ones((coef_num,1)) # intialize theta  

    for i in range(num_iter):  # 这里特别注意，在完成一次循环后，整体更新theta
        #1 每行计算
        '''
        for j in range(len(theta)):
            temp[j] = theta[j] + alpha * np.sum((Y - dot(X, theta)) * X[:,j].reshape(sample_num,1)) / sample_num   
        theta = temp
        print(theta)
        '''
        # 2. 整体矩阵计算梯度
        gradient = np.sum((Y - dot(X, theta))*X)/sample_num
        theta = theta + alpha * gradient
        print(theta)
    return theta



res = LR_gradient([1,2,3],[2,1,2])
