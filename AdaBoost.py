#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:33:49 2020

@author: xinxinma
"""

#%%
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_X_y,check_array,check_is_fitted,assert_all_finite
from sklearn.base import BaseEstimator, ClassifierMixin
#%%
def sgn(x):
    if type(x) != int:
        x[x>=0]=1
        x[x<0]=-1
    elif x>=0:
        x=1
    else:
        x=-1
    return x
#%%
""" IML implementation of an Adaboost Classifier.
    An implementation of the Adaboost algorithm for classification using decision stubs as the weak learner with n_estimators
    parameter. Weights are initalized as uniform, 1/N and learning rate alpha is calculated using 1/2ln(1-err/err) at each
    estimator timestep.
    Parameters
    ----------
    n_estimators : integer, optional (default=50)
        The maximum number of estimators at which boosting is terminated.
    weights : list, optional (default=uniform)
      The distribution of weights for each example
      
    Attributes
    ----------
    weak_learners : 3-tuple of weak_learner stub definition of - [feature stub column index, stub split value, greater then or less
    then equality tag]
    classes_ : array of shape = [n_classes]
        The classes labels.
    alphas : array of floats
        Weights for each estimator in the boosted ensemble.
"""
class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    def __init__ (self, n_clf = 50 , base_estimator = None, n_iter = 30, learning_rate= 1):
        
        self.n_clf = n_clf
        self.base_estimator = base_estimator 
        self.n_iter = n_iter
        self.learning_rate = learning_rate
    
    def fit (self, X, y):
        y =np.where(y==0,-1,y)
        n = X.shape[0]    
        self.w = np.ones(n)/n  # design an equal initial weight 
    
        # step 1: bootstraping and classifer
        
        #set an empty list for classifier
        # initialisation 
        self.classifier = []
        self.y_predict = np.zeros((n, self.n_clf))
        
        for i in range(self.n_clf):
            np.random.seed(i)
            rows = np.random.choice(n,n)
            X_train_bootstrap = X[rows, :]
            y_train_bootstrap = y[rows]
            
            if self.base_estimator == None:
                
                #fit and predict weak classifier
  
                clf=DecisionTreeClassifier(max_depth = 1, class_weight = "balanced")# use decision stump
                clf.fit(X_train_bootstrap, y_train_bootstrap)
                self.classifier.append(clf)
                
                #fil predict result into empty list
                y_pred=clf.predict(X)  # each classifer predict result
                self.y_predict[:,i]=y_pred  # i = 1,2,3....30 个classifiers 
                
            if self.base_estimator == "logit":
                logit=LogisticRegression(class_weight = "balanced")# base estimator as logistic regression
                logit.fit(X_train_bootstrap, y_train_bootstrap)
                self.classifier.append(logit)
                pass
        
        # Step 2: Ensemble
        self.clf = []
        self.pred = np.zeros((n,self.n_iter))
        self.err = np.zeros ((1, self.n_iter))
        self.alpha = np.zeros((1,self.n_iter))
        self.ensemble_pred = np.zeros((n,))
        
        miss = 0.5*np.abs(self.y_predict - y.reshape(-1,1))
        for i in range (self.n_iter):
       
            #error 
            err = np.dot(self.w, miss) 
            err = err * self.learning_rate
            #err = self.w.dot(self.y_predict !=y)
            # set a stopping criteria, if err<0.5 is false， it will stop iteration
            if np.sum(err<0.5)==0:
                break # stop if no good classifier remain 
            
            #Step 2.2: pick smallest misclassfication rate, and corresponding classfier
            
            err_min = np.min(err)
            index = np.argmin(err)
            pred = self.y_predict[:, index]
            clf = self.classifier[index]
            self.clf.append(clf)
            self.err[:,i]=err_min  # self.err[:,i]=err_min
            self.pred[:,i] = pred
          
            # voting power alpha 
            alpha = 0.5*(np.log(1-err_min)-np.log(err_min))  # Boost voting power
            self.alpha[:,i] = alpha
            
            
            # update ensemble model's prediction
            self.ensemble_pred = (self.pred *self.alpha).sum(axis = 1)
            
            # additional stopping criteria 
            if np.sum(np.abs(sgn(self.ensemble_pred)-y)) ==0:
                break
            # update data weight
            self.w[y== pred]/=(2-2*err_min)# /=： c/=a :c=c/a
            self.w[y!= pred]/=(2*err_min)
            self.is_fitted_ = True
            
        return self
            
            
    def predict(self,X):
        assert_all_finite(X)
        check_is_fitted(self,'is_fitted_')
        X= check_array(X,accept_sparse=True)
        
        
        n_iteration=len(self.clf)#ensemble the number of choosen classifier
        self.pred1=np.zeros((X.shape[0],n_iteration))
        self.ensemble_pred1=np.zeros((X.shape[0],))
        
        for i in range(n_iteration):
            pred1=self.clf[i].predict(X)
            self.pred1[:,i] = pred1
            self.ensemble_pred1 = (self.pred1*self.alpha[:,:n_iteration]).sum(axis = 1)
        result = sgn(self.ensemble_pred1)
        return np.where(result ==-1, 0, result)
#%%
from util import check_estimator_adaboost
check_estimator_adaboost(AdaBoostClassifier)