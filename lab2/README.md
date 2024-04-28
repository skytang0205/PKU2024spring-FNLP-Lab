# Dependency

所有代码都是基于Jupyter Notebook平台用python语言，problem1的所有代码在bpe.ipynb文件中，problem2代码在test.ipynb文件中，problem1不需要安装其他库，problem2需要transformers库，所用模型已经包括在文件夹中，不许额外下载。

# Code Structure 
## Problem 1

首先读入bpe-training-data.txt中的训练数据，将句子转化为单词列表，再将每个单词转化为字符列表，在每个单词最后加入'<\s>'字符。

建立词汇表vocabulary，先统计数据中出现的所有字符，和出现次数。

遍历训练集中所有单词的所有字符，用字典bigram_dict统计出现的所有bigram的出现次数，找出出现频率最高的bigram，若其出现频率超过1，加入encode_list中。将训练集中对应的bigram合并为一个token，并更新vocabulary中的出现次数。

将vocabulary写入vocabulary.txt文件，将encode_list写入encode_bigram文件中记录。

按照相同方法预处理bpe-testing-article.txt中的数据。按照encode_list中的顺序，将相应的bigram合并成一个token。最后统计token数量和unknown字符数量。

## Problem 2

在test_text.txt文件和text_text_chinese文件中，我分别准备了一段含有unknown字符的英文句子和中文翻译，以及用电话号码测试长数字的分词。

然后基于transformers库中的AutoTokenizer方法，分别对bert，GPT2，T5，XLM，XLnet，Qwen，Llama等模型进行了分词测试。这些模型都已经包含在文件夹中。