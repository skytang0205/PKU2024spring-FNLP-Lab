import pandas as pd
import numpy as np
import math
import string
import heapq
import sys

data=np.array(pd.read_csv("./ag_news_csv/train.csv"))
with open("./ag_news_csv/classes.txt", 'r') as file:
    class_names=file.read().split()
    print(class_names)

data_num=data.shape[0]
class_num=len(class_names)
print(data.shape,class_num)
largest_feature_num=int(sys.argv[1])

def sentence_process(passage):
    word_list=(passage.lower().translate(str.maketrans(string.punctuation,len(string.punctuation)*' '))).split()
    for i in range(len(word_list)):
        if word_list[i].isdigit():
            word_list[i]='DIGITAL'
    return word_list        

print(sentence_process(data[0][2]))

total_tf_dict={}
total_idf_dict={}
total_tf_idf_dict={}
class_token_num=[0]*class_num

processed_data=[]
for couple in data:
    ##if(count>10000):break
    i=couple[0]-1
    title=sentence_process(couple[1])
    sentence=sentence_process(couple[2])
    couple_words=[]
    for word in title+sentence:
        if word not in total_tf_dict:
            total_tf_dict[word]=[0]*class_num
        total_tf_dict[word][i]+=1
        if word not in couple_words:
            couple_words.append(word)
    for word in couple_words:
        if word not in total_idf_dict:
            total_idf_dict[word]=0
        total_idf_dict[word]+=1
    class_token_num[i]+=len(title+sentence)

print(total_tf_dict['DIGITAL'],len(total_tf_dict))
print(total_idf_dict['DIGITAL'],len(total_idf_dict))
print(class_token_num)

for word in total_tf_dict:
    word_tf_idf=total_tf_dict[word]
    word_idf=math.log(1+data_num/(total_idf_dict[word]+1))+1
    for i in range(class_num):
        word_tf_idf[i]=word_tf_idf[i]/class_token_num[i]*word_idf
    total_tf_idf_dict[word]=word_tf_idf


top_k_items=[]
top_k_keys=[]
for i in range(class_num):
    top_k_items.append(heapq.nlargest(largest_feature_num, total_tf_idf_dict.items(), key=lambda x: x[1][i]))
    top_k_keys.append([item[0] for item in top_k_items[i]])

print("-"*20)
print("100% word tfidf feature:")
print(top_k_keys)

total_tf_dict_1={}
total_tf_idf_dict_1={}

for couple in data:
    ##if(count>10000):break
    i=couple[0]-1
    title=sentence_process(couple[1])
    sentence=sentence_process(couple[2])
    couple_words=[]
    sentence_tf_dict={}
    for word in title+sentence:
        if word not in sentence_tf_dict:
            sentence_tf_dict[word]=0
        sentence_tf_dict[word]+=1
        if word not in couple_words:
            couple_words.append(word)
    sentence_top_k_items=heapq.nlargest(int(len(title+sentence)/4*3), sentence_tf_dict.items(), key=lambda x: x[1]*math.log(1+data_num/(total_idf_dict[x[0]]+1))+1)
    
    for item in sentence_top_k_items:
        if item[0] not in total_tf_dict_1:
            total_tf_dict_1[item[0]]=[0]*class_num
        total_tf_dict_1[item[0]][i]+=item[1]


for word in total_tf_dict_1:
    word_tf_idf=total_tf_dict_1[word]
    word_idf=math.log(1+data_num/(total_idf_dict[word]+1))+1
    for i in range(class_num):
        word_tf_idf[i]=word_tf_idf[i]/class_token_num[i]*word_idf
    total_tf_idf_dict_1[word]=word_tf_idf


top_k_items_1=[]
top_k_keys_1=[]
for i in range(class_num):
    top_k_items_1.append(heapq.nlargest(largest_feature_num, total_tf_idf_dict_1.items(), key=lambda x: x[1][i]))
    top_k_keys_1.append([item[0] for item in top_k_items_1[i]])


print("-"*20)
print("75% word tfidf feature:")
print(top_k_keys_1)

total_tf_dict_2={}
total_tf_idf_dict_2={}

for couple in data:
    ##if(count>10000):break
    i=couple[0]-1
    title=sentence_process(couple[1])
    sentence=sentence_process(couple[2])
    couple_words=[]
    sentence_tf_dict={}
    for word in title+sentence:
        if word not in sentence_tf_dict:
            sentence_tf_dict[word]=0
        sentence_tf_dict[word]+=1
        if word not in couple_words:
            couple_words.append(word)
    sentence_top_k_items=heapq.nlargest(int(len(title+sentence)/2), sentence_tf_dict.items(), key=lambda x: x[1]*math.log(1+data_num/(total_idf_dict[x[0]]+1))+1)
    
    for item in sentence_top_k_items:
        if item[0] not in total_tf_dict_2:
            total_tf_dict_2[item[0]]=[0]*class_num
        total_tf_dict_2[item[0]][i]+=item[1]


for word in total_tf_dict_2:
    word_tf_idf=total_tf_dict_2[word]
    word_idf=math.log(1+data_num/(total_idf_dict[word]+1))+1
    for i in range(class_num):
        word_tf_idf[i]=word_tf_idf[i]/class_token_num[i]*word_idf
    total_tf_idf_dict_2[word]=word_tf_idf


top_k_items_2=[]
top_k_keys_2=[]
for i in range(class_num):
    top_k_items_2.append(heapq.nlargest(largest_feature_num, total_tf_idf_dict_2.items(), key=lambda x: x[1][i]))
    top_k_keys_2.append([item[0] for item in top_k_items_2[i]])


print("-"*20)
print("50% word tfidf feature:")
print(top_k_keys_2)

total_tf_dict_3={}
total_tf_idf_dict_3={}

for couple in data:
    ##if(count>10000):break
    i=couple[0]-1
    title=sentence_process(couple[1])
    sentence=sentence_process(couple[2])
    couple_words=[]
    sentence_tf_dict={}
    for word in title+sentence:
        if word not in sentence_tf_dict:
            sentence_tf_dict[word]=0
        sentence_tf_dict[word]+=1
        if word not in couple_words:
            couple_words.append(word)
    sentence_top_k_items=heapq.nlargest(int(len(title+sentence)/4), sentence_tf_dict.items(), key=lambda x: x[1]*math.log(1+data_num/(total_idf_dict[x[0]]+1))+1)
    
    for item in sentence_top_k_items:
        if item[0] not in total_tf_dict_3:
            total_tf_dict_3[item[0]]=[0]*class_num
        total_tf_dict_3[item[0]][i]+=item[1]


for word in total_tf_dict_3:
    word_tf_idf=total_tf_dict_3[word]
    word_idf=math.log(1+data_num/(total_idf_dict[word]+1))+1
    for i in range(class_num):
        word_tf_idf[i]=word_tf_idf[i]/class_token_num[i]*word_idf
    total_tf_idf_dict_3[word]=word_tf_idf


top_k_items_3=[]
top_k_keys_3=[]
for i in range(class_num):
    top_k_items_3.append(heapq.nlargest(largest_feature_num, total_tf_idf_dict_3.items(), key=lambda x: x[1][i]))
    top_k_keys_3.append([item[0] for item in top_k_items_3[i]])

print("-"*20)
print("25% word tfidf feature:")
print(top_k_keys_3)


all_feature=[]
for i in range(class_num):
    for word in top_k_keys_2[i]:
        if len(word)>1 and word not in all_feature:
            all_feature.append(word)
    for word in top_k_keys_3[i]:
        if len(word)>1 and word not in all_feature:
            all_feature.append(word)
        

print("final word tfidf feature:")
print(len(all_feature))
print(all_feature)

file_path = './features&model/feature.txt'  # 指定文件路径及名称
with open(file_path, 'w') as file:
    for word in all_feature:
        file.write(word+' ')

print('文件写入完成。')

