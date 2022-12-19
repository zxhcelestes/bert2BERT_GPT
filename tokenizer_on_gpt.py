import os.path

from transformers import GPT2Tokenizer
import pickle
import json


def get_sst_tokenizer():
    if not (os.path.exists("./SST-2/vocab.json") and os.path.exists("./SST-2/merges.txt")):
        build_vocab()
    return GPT2Tokenizer("./SST-2/vocab.json", "./SST-2/merges.txt")


def build_vocab():
    f = open('./SST-2/vocab.pkl', 'rb')
    data = pickle.load(f)
    with open("SST-2/merges.txt", "w", encoding='utf-8') as f:
        f.write("<|endoftext|>\n")
        f.write("<|endoftext|>\n")
        f.write("<|endoftext|>\n")
        # f.write("[MASK]\n")
        for key in data:
            f.write(key + "\n")
    with open("SST-2/merges.txt", 'r', encoding='utf-8') as f:  # 打开txt文件
        cnt=0
        d = {}
        for line in f:
            d[line.rstrip('\n')] = cnt  # line表示txt文件中的一行，将每行后面的换行符去掉，并将其作为字典d的content键的值
            cnt+=1
        with open('SST-2/vocab.json', 'w', encoding='utf-8') as file:  # 创建一个json文件，mode设置为'a'
            json.dump(d, file,ensure_ascii=False)  # 将字典d写入json文件中，并设置ensure_ascii = False,主要是因为汉字是ascii字符码,若不指定该参数，那么文字格式就会是ascii码
            # file.write('\n')


if __name__ == '__main__':
    build_vocab()
