from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

with open('test_text.txt', 'r') as file:
    # 读取文件内容
    data = file.read()

tokenized_data = tokenizer.tokenize(data)

print(tokenized_data)