import pandas as pd
import numpy as np
import math
import string
import heapq
import random
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

data=np.array(pd.read_csv("./ag_news_csv/train.csv"))
with open("./ag_news_csv/classes.txt", 'r') as file:
    class_names=file.read().split()

data_num=data.shape[0]
class_num=len(class_names)
print(data.shape,class_num)
largest_feature_num=int(sys.argv[1])

lr=1e-2
wr=1e-4
with open('./features&model/feature.txt', 'r') as file:
    features=file.read().split()
    ##print(feature)
feature_word=np.array([features]*class_num)
print(feature_word.shape)
largest_feature_num=min(largest_feature_num,feature_word.shape[1])

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

#测试用函数
def test_loglinear_model(model, data_features, labels):    
    pred_labels=model.test(data_features)
    print("accuracy_score: ",accuracy_score(labels, pred_labels))
    print("f1_score: ",f1_score(labels, pred_labels, average=None),f1_score(labels, pred_labels, average='macro'))
    file_path = './result/acc.txt'  # 指定文件路径及名称
    with open(file_path, 'w') as file:
        file.write('train accuracy_score: ')
        file.write(str(accuracy_score(labels, pred_labels)))        
        file.write('\n')
        file.write('train f1_score: ')
        for i in f1_score(labels, pred_labels, average=None):
            file.write(str(i)+' ')
        file.write('\n')
        file.write(str(f1_score(labels, pred_labels, average='macro')))
        file.write('\n')
    print('文件写入完成。')

#画图用函数
def show_plt(all_avg_loss):
    iterations = range(1, len(all_avg_loss) + 1)
    plt.plot(iterations, all_avg_loss, 'b-')
    plt.title('Loss Curve')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')

    plt.grid(True)
    
# 训练Log-Linear模型
def train_loglinear_model(model, data, num_epochs, learning_rate, weight_rate):
    data_features,labels=trans_feature(data,model.feature_word)

    all_avg_loss=[]
    for epoch in range(num_epochs):
        total_loss = 0.0
        count=0
        for i in range(data_features.shape[0]):
            count+=1
            #if(count%int(data_num/100)==0):print("trans got {}0%",count/int(data_num/10))
            sentence_feature=data_features[i]
            label = labels[i]

            model.update_weights(sentence_feature, label, learning_rate, weight_rate)

            predicted = model.predict(sentence_feature)
            loss = -math.log(predicted[label])
            total_loss += loss

        # 打印每个epoch的平均损失
        avg_loss = total_loss / data_features.shape[0]
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss}")
        all_avg_loss.append(avg_loss)

    show_plt(all_avg_loss)
    test_loglinear_model(model, data_features, labels)



    return model


sample_num = int(data_num/10) 
sample_list = [i for i in range(data_num)]
sample_list = random.sample(sample_list, sample_num)
 
pre_train_data= data[sample_list,:]
num_epochs = 15


model = LogLinearModel(feature_word)
#预训练
model = train_loglinear_model(model, pre_train_data, num_epochs*3, lr, wr)
plt.savefig( './result/pretrain1.png') 

features_weights = np.sum(model.weights*model.weights,axis=0)
sorted_indices = np.argsort(features_weights)
real_features = (np.array(features)[sorted_indices])[:largest_feature_num]

file_path = './features&model/real_feature.txt'  # 指定文件路径及名称
with open(file_path, 'w') as file:
    for word in real_features:
        file.write(word+' ')

print('文件写入完成。')
real_feature_word = np.array([real_features.tolist()]*class_num)

real_model = LogLinearModel(real_feature_word)
#real_model.bias=model.bias*real_feature_word.shape[0]/feature_word.shape[0]
#real_model.weights=(model.weights[:,sorted_indices])[:,:largest_feature_num]
real_model = train_loglinear_model(real_model, pre_train_data, num_epochs*3, lr, wr)
plt.savefig( './result/pretrain2.png') 
#训练


real_model = train_loglinear_model(real_model, data, num_epochs, lr, wr)
plt.savefig( './result/train.png') 

file_path = './features&model/model.txt'  # 指定文件路径及名称
with open(file_path, 'w') as file:
    for i in range(class_num):
        for weight in real_model.weights[i]:
            file.write(str(weight)+' ')
        file.write('\n')
    for bias in real_model.bias:
        file.write(str(bias)+' ')
print('文件写入完成。')
    




