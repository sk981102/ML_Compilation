import numpy as np
import pandas as pd
from math import pi
from math import sqrt
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score


# NAIVE BAYES algorithm implementation to predict Dementia based on 2 features: RightHippoVol and LeftHippoVol 

def calc_prob(x, mean, std):
	exponent = np.exp(-((x-np.mean)**2 / (2 * np.std**2 )))
	return (1 / (np.sqrt(2 * pi) * np.std)) * exponent


def summarize(train):
    summary=dict()
    for column in train.columns: 
        col_mean=np.mean(train[column])
        col_std=np.std(train[column])
        col_len=len(train[column])
        summary[column]=[col_mean,col_std,col_len]
    return summary


def naive_train(train):
    summary = {0:{},1:{}}
    
    when0=train[train["Dementia"]==0]
    when1=train[train["Dementia"]==1]
    
    
    summary[0]=summarize(when0[["RightHippoVol","LeftHippoVol"]])
    summary[1]=summarize(when1[["RightHippoVol","LeftHippoVol"]])
    
    total_rows=len(train.index)
    
 

    return summary

def naive_predict_row(summary,row,total_rows):

    probs=dict()
    
    for class_val, class_sum in summary.items():
        instance=len(train[train["Dementia"]==class_val].index)
        
        probs[class_val]=instance/float(total_rows)
        
        
        for i,list in (class_sum.items()):
            mean=list[0]
            std=list[1]
            num=calc_prob(row[i],mean,std)
            probs[class_val]*=num
            
    
    prob_dem=probs[1]/(probs[1]+probs[0])
    
    if(prob_dem>0.5):
        return 1
    else:
        return 0
    
def naive_predict(data,summary):
    total_num=len(data.index)
    predictions=[]
    
    for i in range(total_num):
        predictions.append(naive_predict_row(summary,data.iloc[i,],total_num))
    return predictions

def accuracy(data,predictions):
    return np.mean(data["Dementia"]==predictions)

if __name__ == "__main__":
    np.random.seed(42)
    
    #reading and splitting test and train
    data=pd.read_csv("data/Thoughts.docx.csv",sep=',')
    tr=data[data["TrainData"]==1]
    te=data[data["TrainData"]==0]
    train=tr[["RightHippoVol","LeftHippoVol","Dementia"]]
    test=te[["RightHippoVol","LeftHippoVol","Dementia"]]
    
    #training
    summ=naive_train(train)
    
    #predicting rows and calculating accuracy on train data
    total_num=len(train.index)
    print(accuracy(train,naive_predict(train,summ))) #0.7767857142857143
    
    #predicting rows and calculating accuracy on test data
    print(accuracy(test,naive_predict(test,summ))) #0.7079646017699115
    
    #Let's test on standard library
    X_train=train[["RightHippoVol","LeftHippoVol"]]
    y_train=train["Dementia"]

    X_test=test[["RightHippoVol","LeftHippoVol"]]
    y_test=test["Dementia"]
    
    #train
    gnb = GaussianNB() 
    gnb.fit(X_train,y_train)
    
    #train prediction
    y_pred = gnb.predict(X_train)
    print(accuracy_score(y_train, y_pred)) # train: 0.7767857142857143
    
    #test prediction
    y_pred = gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred)) # test: 0.7079646017699115

    # the same! implementation successful 
    
    
    
    

