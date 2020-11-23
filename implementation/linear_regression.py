import pandas as pd
import numpy as np

def rmse(y, y_pred):
    rmse = np.sqrt(sum((y - y_predict) ** 2) / len(y))
    return rmse

def r2(y,y_predict):
    sst=np.sum((y-np.mean(y))**2)
    sse=np.sum((y-y_predict)**2)
    return 1-(sse/sst)

# NORMAL EQUATION Way
class MyNormalLinearRegression:
    def __init__ (self,i):
        """
        Initializing the regressor
        """
        self.theta=np.zeros(i,float)[:,np.newaxis]; #parameter vector; random
    
    def fitUsingNormalEquation(self,X_train,y_train):
        """
        Training using the Closed Form Equation
        """
        X_b=np.c_[np.ones(len(X_train)),X_train] #adding additional column
        theta_best=np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
        self.theta=theta_best
    
    def predict(self,X_test):
        """
        Predicting the label
        """
        X_test = np.c_[np.ones((len(X_test),1)), X_test]
        y_predict = np.dot(X_test,self.theta)
        
        return y_predict
