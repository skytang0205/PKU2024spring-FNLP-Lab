#!/bin/bash

# 定义参数
int_feature_detect_num=1100
int_feature_num=4500

# 定义一个变量
arg=$1

if [ $arg -ge 2 ]
then
    echo "feature extraction and training"
    python get_feature.py "$int_feature_detect_num"
    python training.py "$int_feature_num"
fi
echo "evaluation"
python evaluation.py