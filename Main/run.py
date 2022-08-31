#In this project I will try to implment some of the machine learning algorithms using basic libraries such as numpy.
#The goal of this project is to practice the implementation techniques of machine learning models and go in detail on how each model works and
#try to make some optimizations for them.
#This is a great opprotunity to practice what I've learnt so far about the machine learning algorithms, to be more familiar about how the model
#work in the background and how I can optimize them later to give better results.
#In this project I will focus mainly on implementing 3 models which are : Linear Regression, Logistic Regression and Decision Trees.
#I may later try to deploy it with fast API and try to make a visualization to make it look cool.
#The first version of this project should be an implementation of the linear regression model and it should take me about a week to get everything working

import models.LinearRegression.model as lr
import numpy as np
from sklearn.linear_model import Ridge,Lasso
def main():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3],[4, 5], [3, 2], [4, 2], [4, 1],[1, 4], [3, 4], [5, 2], [5, 1]])
    y = np.dot(X, np.array([120.5, -2.6])) + 0.08
    initial_weights=np.array([0.,0.])
    model=lr.LinearRegression()
    model.fit(X,y,initial_weights,tolerance=1e-8)
    W,b=model.get_coef()

    print(W)
    print(b)  

    model_sk_learn_Ridge=Ridge(alpha=0.1,tol=1e-4)
    model_sk_learn_Ridge.fit(X,y)
    print(f'Params of Sk Learn Ridge: {model_sk_learn_Ridge.coef_}')
    
    model_ridge=lr.LinearRegression(l2=0.1)
    model_ridge.fit(X,y,initial_weights,step_size=0.001,tolerance=20)

    W,b=model_ridge.get_coef()
    print(f'Params my ridge: {W}')

    model_sk_learn_Lasso=Lasso(alpha=0.1,tol=1e-4)
    model_sk_learn_Lasso.fit(X,y)
    print(f'Params of Sk Learn Lasso: {model_sk_learn_Lasso.coef_}')
    
    model_lasso=lr.LinearRegression(l1=0.1)
    model_lasso.fit(X,y,initial_weights,step_size=0.001,tolerance=1e-4,max_steps=10000)

    W,b=model_lasso.get_coef()
    print(f'Params my lasso: {W}')
  





if __name__ =="__main__":
    main()
    