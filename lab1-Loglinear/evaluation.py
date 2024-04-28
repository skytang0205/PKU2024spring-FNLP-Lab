import pandas as pd
import numpy as np
import math
import string
import heapq
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data=np.array(pd.read_csv("./ag_news_csv/test.csv"))
with open("./ag_news_csv/classes.txt", 'r') as file:
    class_names=file.read().split()

data_num=data.shape[0]
class_num=len(class_names)
print(data.shape,class_num)

## pre-training ##

with open('./features&model/real_feature.txt', 'r') as file:
    feature=file.read().split()
    ##print(feature)
feature_word=np.array([feature]*class_num)
print(feature_word.shape)

def sentence_process(passage):
    word_list=(passage.lower().translate(str.maketrans(string.punctuation,len(string.punctuation)*' '))).split()
    for i in range(len(word_list)):
        if word_list[i].isdigit():
            word_list[i]='DIGITAL'
    return word_list   

def trans_feature(data,feature_word):
    data_features=[]
    labels=[]
    count=0
    for couple in data:
        count+=1
        if(count%int(data.shape[0]/10)==0):print("trans got {}0%".format(int(count/int(data.shape[0]/10))))
        ##if(count>10000):break
        i=couple[0]-1
        labels.append(i)
        sentence_feature=np.zeros(feature_word.shape,dtype=np.float64)

        title=sentence_process(couple[1])
        sentence=sentence_process(couple[2])        
        for word in title+sentence:
            for i in range(class_num):
                if word in feature_word[i]:
                    sentence_feature[i][np.where(feature_word[i]==word)[0]]+=1.0
        data_features.append(sentence_feature)
    return np.array(data_features),np.array(labels)

class LogLinearModel:
    def __init__(self, feature_word):
        self.feature_word=feature_word
        self.num_features = feature_word.shape
        self.weights = np.zeros(self.num_features,dtype=np.float64)
        self.bias = np.zeros(self.num_features[0],dtype=np.float64)

    def predict(self, features):
        temp=np.sum(features*self.weights,1)+self.bias
        return np.exp(temp)/np.sum(np.exp(temp))

    def update_weights(self, sentence_feature, label, learning_rate, weight_rate):
        k=np.zeros(class_num,dtype=np.float64)
        k[label]=1.0
        temp=self.predict(sentence_feature)
        weight_loss = (sentence_feature.T*(temp-k)).T
        bias_loss =temp-k
        self.weights -= learning_rate * (weight_loss + weight_rate*self.weights)
        self.bias -= learning_rate * bias_loss
    
    def test(self, data_features):
        pred_labels=[]
        for sentence_feature in data_features:
            temp=self.predict(sentence_feature)
            pred_labels.append(np.argmax(temp))
        return pred_labels

def test_loglinear_model(model, data_features, labels):    
    pred_labels=model.test(data_features)
    confusion_matrix=np.zeros((class_num,class_num))
    num=labels.shape[0]
    for i in range(num):
        confusion_matrix[labels[i]][pred_labels[i]]+=1
    print(confusion_matrix)
    right=0.0
    for i in range(class_num):
        right+=confusion_matrix[i][i]
    f1=[]
    for i in range(class_num):
        k=0.0
        t=0.0
        for j in range(class_num):
            k+=confusion_matrix[i][j]
            t+=confusion_matrix[j][i]
        k=confusion_matrix[i][i]/k
        t=confusion_matrix[i][i]/t
        f1.append(2*(k*t)/(k+t))
    print("my accuracy_score: ",right/num)
    print("my f1_score: ",f1,np.mean(f1))
    print("skl_accuracy_score: ",accuracy_score(labels, pred_labels))
    print("f1_score: ",f1_score(labels, pred_labels, average=None),f1_score(labels, pred_labels, average='macro'))
    file_path = './result/acc.txt'  # 指定文件路径及名称
    with open(file_path, 'a') as file:
        file.write('test accuracy_score: ')
        file.write(str(accuracy_score(labels, pred_labels)))        
        file.write('\n')
        file.write('test f1_score: ')
        for i in f1_score(labels, pred_labels, average=None):
            file.write(str(i)+' ')
        file.write('\n')
        file.write(str(f1_score(labels, pred_labels, average='macro')))
        file.write('\n')
    print('文件写入完成。')

model = LogLinearModel(feature_word)
with open('./features&model/model.txt', 'r') as file:
    lines = file.readlines()

for i in range(class_num):
    line=lines[i].split()
    for j in range(len(line)):
        model.weights[i][j]=float(line[j])
line=lines[class_num].split()
for i in range(len(line)):
    model.bias[i]=float(line[i])

data_features, labels=trans_feature(data, model.feature_word)
test_loglinear_model(model, data_features, labels)
