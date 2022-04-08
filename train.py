import numpy as np
import random
import torch
import matplotlib.pylab as plt
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
from functools import partial  #partial()函数可以用来固定某些参数值，并返回一个新的callable对象
import pdb
# 数据格式调整，将原先每行是每个字的标注形式，修改为每行是每句话的标注形式，相邻字（标注）之间，采用符号'\002'进行分隔
def format_data(source_filename, target_filename):
    datalist = []
    # 读取source_filename所有数据到lines中，每个元素是字标注
    with open(source_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 逐个处理每个字标注
    words = ''
    labels = ''
    # 当前处理的是否为每句话首字符，0：是，1：不是
    flag = 0
    for line in lines:
        # 空行表示每句话标注的结束
        if line == '\n':
            # 连接文本和标注结果
            item = words + '\t' + labels + '\n'
            # print(item)
            # 添加到结果列表中
            datalist.append(item)
            # 重置文本和标注结果
            words = ''
            labels = ''
            flag = 0
            continue
            # 分离出字和标注
        word, label = line.strip('\n').split(' ')
        # 不是每句话的首字符
        if flag == 1:
            # words/labels非空，和字/标签连接时需要添加分隔符'\002'
            words = words + '\002' + word
            labels = labels + '\002' + label
        else:  # 每句话首字符，words/labels为空，和字/标签连接时不需要添加分隔符'\002'
            words = words + word
            labels = labels + label
            flag = 1  # 修改标志
    with open(target_filename, 'w', encoding='utf-8') as f:
        #pdb.set_trace()
        #将转换结果写入文件target_filename
        lines=f.writelines(datalist)
    print(f'{source_filename}文件格式转换完毕，保存为{target_filename}')



#构建Label标签表
# 提取文件source_filename1和source_filename2的标签类型，保存到target_filename
def gernate_dic(source_filename1, source_filename2, target_filename):
    # 标签类型列表初始化为空
    data_list = []

    # 读取source_filename1所有行到lines中，每行元素是单字和标注
    with open(source_filename1, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 处理每行数据（单字+‘ ’+标注）
    for line in lines:
        # 数据非空
        if line != '\n':
            # 提取标注，-1是数组最后1个元素
            dic = line.strip('\n').split(' ')[-1]
            # 不在标签类型列表中，则添加
            if dic + '\n' not in data_list:
                data_list.append(dic + '\n')

    # 读取source_filename2所有行到lines中，每行元素是单字和标注
    with open(source_filename2, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 处理每行数据（单字+‘ ’+标注）
    for line in lines:
        # 数据非空
        if line != '\n':
            # 提取标注，-1是数组最后1个元素
            dic = line.strip('\n').split(' ')[-1]
            # 不在标签类型列表中，则添加
            if dic + '\n' not in data_list:
                data_list.append(dic + '\n')

    with open(target_filename, 'w', encoding='utf-8') as f:
        # 将标签类型列表写入文件target_filename
        lines = f.writelines(data_list)
def MapDataset(listofdata):
    return [listofdata[0],listofdata[1]]

#加载自定义数据集
# 加载数据文件datafiles
def load_dataset(datafiles):
    # 读取数据文件data_path
    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)  # Skip header  #Deleted by WGM
            # 处理每行数据（文本+‘\t’+标注）
            for line in fp.readlines():
                # 提取文本和标注
                words, labels = line.strip('\n').split('\t')
                # 文本中单字和标注构成的数组
                words = words.split('\002')
                labels = labels.split('\002')
                # 迭代返回文本和标注
                yield words, labels


    # 根据datafiles的数据类型，选择合适的处理方式
    if isinstance(datafiles, str):  # 字符串，单个文件名称
        # 返回单个文件对应的单个数据集
        # return MapDataset(list(read(datafiles)))
        return list(read(datafiles))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):  # 列表或元组，多个文件名称
        # 返回多个文件对应的多个数据集
        return [list(read(datafile)) for datafile in datafiles]


# 加载字典文件，文件由单列构成，需要设置value
def load_dict_single(dict_path):
    # 字典初始化为空
    vocab = {}
    # value是自增数值，从0开始
    i = 0
    # 逐行读取字典文件
    for line in open(dict_path, 'r', encoding='utf-8'):
        # 将每行文字设置为key
        key = line.strip('\n')
        # 设置对应的value
        vocab[key] = i
        i += 1
    return vocab

#逐个转换文件
# format_data('./dataset/dev.conll', './dataset/dev.txt')
# format_data(r'./dataset/train.conll', r'./dataset/train.txt')

# 根据训练集和验证集生成dic，保存所有的标签
# gernate_dic('dataset/train.conll', 'dataset/dev.conll', 'dataset/mytag.dic')

# 加载Bert模型需要的输入数据
train_ds, dev_ds = load_dataset(datafiles=(
        './dataset/train.txt', './dataset/dev.txt'))
#加载标签文件，并转换为KV表，K为标签，V为编号（从0开始递增）
label_vocab = load_dict_single('./dataset/mytag.dic')

# print("训练集、验证集、测试集的数量：")
# print(len(train_ds),len(dev_ds))
# print(train_ds[0])
# print(dev_ds[0])
# print(label_vocab)



#数据预处理
#tokenizer：预编码器，label_vocab：标签类型KV表，K是标签类型，V是编码
def convert_example(example,tokenizer,label_vocab,max_seq_length=256,is_test=False):
    #测试集没有标签
    if is_test:
        text = example
    else:#训练集和验证集包含标签
        text, label = example
    #tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    #encode仅返回input_ids
    #encode_plus返回所有编码信息 input_ids’：是单词在词典中的编码 token_type_ids’：区分两个句子的编码（上句全为0，下句全为1） ‘attention_mask’：指定对哪些词进行self-Attention操作
    encoded_inputs = tokenizer.encode_plus(text=text, max_seq_len=None, pad_to_max_seq_len=False, return_length=True)
    #pdb.set_trace()
    #获取字符编码（'input_ids'）、类型编码（'token_type_ids'）、字符串长度（'seq_len'）
    input_ids = encoded_inputs["input_ids"]
    segment_ids = encoded_inputs["token_type_ids"]
    seq_len = len(input_ids)-2

    if not is_test:#训练集和验证集
        #[CLS]和[SEP]对应的标签均是['O']，添加到标签序列中
        label = ['O'] + label + ['O']
        #生成由标签编码构成的序列
        label = [label_vocab[x] for x in label]
        return input_ids, segment_ids, seq_len, label
    else:#测试集，不返回标签序列
        return input_ids, segment_ids, seq_len


# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
# b. 导入配置文件
model_config = BertConfig.from_pretrained("bert-base-chinese")
#对训练集和测试集进行编码
MODEL_PATH = './dataset/model/bert-base-chinese/'

#functools.partial()的功能：预先设置参数，减少使用时设置的参数个数
#使用partial()来固定convert_example函数的tokenizer, label_vocab, max_seq_length等参数值
trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab, max_seq_length=128)


# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(MODEL_PATH, config = model_config)

print(tokenizer.encode('我不喜欢你的猫'))
#对训练集和测试集进行编码
for train in train_ds:
    print(trans_func(train))
# train_ds.map(trans_func)


for dev in dev_ds:
    print(trans_func(dev))
print ([train_ds[0]])