import numpy as np
import random
import torch
import matplotlib.pylab as plt
import torchmetrics
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction,  BertForQuestionAnswering
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from functools import partial  # partial()函数可以用来固定某些参数值，并返回一个新的callable对象
import pdb
from transformers import BertForTokenClassification

"""
# 1.数据格式调整，将原先每行是每个字的标注形式，修改为每行是每句话的标注形式，相邻字（标注）之间，采用符号'\002'进行分隔
"""
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
        # pdb.set_trace()
        # 将转换结果写入文件target_filename
        lines = f.writelines(datalist)
    print(f'{source_filename}文件格式转换完毕，保存为{target_filename}')


"""
# 2.构建Label标签表
"""
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
    return [listofdata[0], listofdata[1]]


"""
# 3.加载自定义数据集
"""
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


# 逐个转换文件
# format_data('./dataset/dev.conll', './dataset/dev.txt')
# format_data(r'./dataset/train.conll', r'./dataset/train.txt')

# 根据训练集和验证集生成dic，保存所有的标签
# gernate_dic('dataset/train.conll', 'dataset/dev.conll', 'dataset/mytag.dic')

"""
# 4.加载Bert模型需要的输入数据
"""
train_ds, dev_ds = load_dataset(datafiles=(
    './dataset/train.txt', './dataset/dev.txt'))
# 加载标签文件，并转换为KV表，K为标签，V为编号（从0开始递增）
label_vocab = load_dict_single('./dataset/mytag.dic')


# print("训练集、验证集、测试集的数量：")
# print(len(train_ds),len(dev_ds))
# print(train_ds[0])
# print(dev_ds[0])
# print(label_vocab)


"""
# 5.数据预处理
"""
# tokenizer：预编码器，label_vocab：标签类型KV表，K是标签类型，V是编码
def convert_example(example, tokenizer, label_vocab, max_seq_length=256, is_test=False):
    # 测试集没有标签
    if is_test:
        text = example
    else:  # 训练集和验证集包含标签
        text, label = example
    # tokenizer.encode方法能够完成切分token，映射token ID以及拼接特殊token
    # encode仅返回input_ids
    # encode_plus返回所有编码信息 input_ids’：是单词在词典中的编码 token_type_ids’：区分两个句子的编码（上句全为0，下句全为1） ‘attention_mask’：指定对哪些词进行self-Attention操作
    encoded_inputs = tokenizer.encode_plus(text=text, max_seq_len=None, pad_to_max_seq_len=False, return_length=True)
    # pdb.set_trace()
    # 获取字符编码（'input_ids'）、类型编码（'token_type_ids'）、字符串长度（'seq_len'）
    input_ids = encoded_inputs["input_ids"]
    segment_ids = encoded_inputs["token_type_ids"]
    seq_len = torch.tensor(len(input_ids))

    if not is_test:  # 训练集和验证集
        # [CLS]和[SEP]对应的标签均是['O']，添加到标签序列中
        label = ['O'] + label + ['O']
        # 生成由标签编码构成的序列
        label = [label_vocab[x] for x in label]
        return input_ids, segment_ids, seq_len, label
    else:  # 测试集，不返回标签序列
        return input_ids, segment_ids, seq_len


# a.通过词典导入分词器
tokenizer = BertTokenizer.from_pretrained("./dataset/model/bert-base-chinese/")
# b. 导入配置文件
# model_config = BertConfig.from_pretrained("./dataset/model/bert-base-chinese/")
# 对训练集和测试集进行编码
MODEL_PATH = './dataset/model/bert-base-chinese/'

# functools.partial()的功能：预先设置参数，减少使用时设置的参数个数
# 使用partial()来固定convert_example函数的tokenizer, label_vocab, max_seq_length等参数值
trans_func = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab, max_seq_length=128)
trans_func_test = partial(convert_example, tokenizer=tokenizer, label_vocab=label_vocab, max_seq_length=128,
                          is_test=True)

# 修改配置
# model_config.output_hidden_states = True
# model_config.output_attentions = True
# 通过配置和路径导入模型
# bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)

# 对训练集和测试集进行编码
# print(train_ds[0])
for index, train in enumerate(train_ds):
    train_ds[index] = trans_func(train)
for index, dev in enumerate(dev_ds):
    dev_ds[index] = trans_func(dev)
# 输出转换后的向量
# 向量转单词

# print(train_ds[0])
# print(tokenizer.convert_ids_to_tokens(train_ds[0][0]))


def create_mini_batch(samples):
    tokens_tensors = [torch.tensor(s[0]) for s in samples]
    sql_lens = [s[2] for s in samples]
    label_tensors=None
    if len(samples[0])==4:
        label_tensors = [torch.tensor(s[3]) for s in samples]
    else:
        label_tensors = [torch.tensor([0]*s[2].item()) for s in samples]
    # pad_sequence传入的数据必须的tensor
    # zero pad 到同一序列长度
    one = [0]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    label_tensors = pad_sequence(label_tensors, batch_first=True)


    tokens_tensors = torch.tensor([t + one for t in tokens_tensors.numpy().tolist()])
    label_tensors = torch.tensor([t + one for t in label_tensors.numpy().tolist()])

    # attention masks，将 tokens_tensors 不为 zero padding 的位置设为1
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)  # segment_ids

    return tokens_tensors, masks_tensors, sql_lens, label_tensors


# create_mini_batch(train_ds)
trainloader = DataLoader(train_ds, batch_size=32, collate_fn=create_mini_batch, drop_last=False)
devloader = DataLoader(dev_ds, batch_size=32, collate_fn=create_mini_batch, drop_last=False)


"""
# 6.Bert模型加载和训练
"""
model = BertForTokenClassification.from_pretrained(MODEL_PATH,num_labels=len(label_vocab))
# model.cuda()
# 设置Fine-Tune优化策略
# 计算了块检测的精确率、召回率和F1-score。常用于序列标记任务，如命名实体识别
# 实例化相关metrics的计算对象
model_recall = torchmetrics.Recall(average='macro', num_classes=len(label_vocab.keys()))
model_precision = torchmetrics.Precision(average='macro', num_classes=len(label_vocab.keys()))
model_f1 = torchmetrics.F1Score(average="macro", num_classes=len(label_vocab.keys()))
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss(ignore_index=-1)
# 在Adam的基础上加入了权重衰减的优化器，可以解决L2正则化失效问题
optimizer = torch.optim.Adam(lr=2e-5, params=model.parameters())



#只对二分类问题有用
# def performance_index(predict, target):
#     confusion_matrix = torch.zeros(len(label_vocab.keys()), len(label_vocab.keys()))
#     for p, t in zip(predict.view(-1), target.view(-1)):
#         confusion_matrix[t.long(), p.long()] += 1
#     a_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[0]
#     b_p = (confusion_matrix.diag() / confusion_matrix.sum(1))[1]
#     a_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[0]
#     b_r = (confusion_matrix.diag() / confusion_matrix.sum(0))[1]
#     return a_p, b_p, a_r, b_r


# 评估函数
def evaluate(model, data_loader):
    # 依次处理每批数据
    for input_ids, masks_tensors, lens, labels in data_loader:
        # 单字属于不同标签的概率
        output= model(input_ids=input_ids,attention_mask = masks_tensors,labels =labels)
        # 损失函数的平均值
        loss = output[0]
        # 按照概率最大原则，计算单字的标签编号
        # argmax计算logits中最大元素值的索引，从0开始
        preds = torch.argmax(output[1], axis=-1)

        model_f1.update(preds.flatten(), labels.flatten())
        model_recall.update(preds.flatten(), labels.flatten())
        model_precision.update(preds.flatten(), labels.flatten())
    f1_score = model_f1.compute()
    recall = model_recall.compute()
    precision = model_precision.compute()

    # 清空计算对象
    model_precision.reset()
    model_f1.reset()
    model_recall.reset()
    print("评估准确度: %.6f - 召回率: %.6f - f1得分: %.6f- 损失函数: %.6f" % (precision, recall, f1_score, total_loss))



# 模型训练
global_step = 0
for epoch in range(1000):
    # 依次处理每批数据
    for step, (input_ids, masks_tensors, seq_lens, labels) in enumerate(trainloader,start=1):
        # 单字属于不同标签的概率
        output = model(input_ids=input_ids,attention_mask = masks_tensors,labels =labels)
        total_loss = 0.0
        """
        这一段是输出每一轮循环的准确率
        """
        # 按照概率最大原则，计算单字的标签编号
        preds = torch.argmax(output[1], dim=-1)
        # 推理块个数，标签个数，正确的标签个数
        # n_infer, n_label, n_correct = classification_report(seq_lens, preds, labels)
        # 更新评估的参数
        # metric.update(n_infer.numpy(), n_label.numpy(), n_correct.numpy())
        # 平均化的准确率、召回率、F1值
        # precision, recall, f1_score = metric.accumulate()
        # performance_index(preds, labels)
        # 实例化相关metrics的计算对象
        # 一个batch进行计算迭代


        # model_acc.update(preds, labels)
        recall=model_recall(preds.flatten(), labels.flatten())
        precision=model_precision(preds.flatten(), labels.flatten())
        f1_score = model_f1(preds.flatten(), labels.flatten())
        # model_recall.update(preds.flatten(), labels.flatten())
        # model_precision.update(preds.flatten(), labels.flatten())
        # model_f1.update(preds.flatten(), labels.flatten())
        # total_recall = model_recall.compute()
        loss = output[0]
        total_loss+=loss.item()
        # 损失函数的平均值
        if global_step % 10 == 0:
            pass
            print("训练集的当前epoch:%d - step:%d" % (epoch, step))
            print("训练准确度: %.6f, 召回率: %.6f, f1得分: %.6f- 损失函数: %.6f" % (precision, recall, f1_score, loss))
            # print("训练准确度: %.6f, 召回率: %.6f, f1得分: %.6f" % (precision, recall,  f1_score))

        """
        这一段是backforward 向后传播，优化w
        """
        # 回传损失函数，计算梯度
        loss.backward()
        # 根据梯度来更新参数
        optimizer.step()
        # 梯度置零
        optimizer.zero_grad()
        global_step += 1
    # 计算一个epoch的accuray、recall、precision
    total_recall = model_recall.compute()
    total_precision = model_precision.compute()
    total_f1 = model_precision.compute()

    # 清空计算对象
    model_precision.reset()
    model_f1.reset()
    model_recall.reset()

    # 评估训练模型
    evaluate(model, devloader)
    torch.save(model.state_dict(),
               "./checkpoint/model_%d.pdparams"% (global_step))

# 模型存储
# !mkdir bert_result
model.save_pretrained('./bert_result')
tokenizer.save_pretrained('./bert_result')

"""
#7.模型加载与处理数据
"""
# 加载测试数据
def load_testdata(datafiles):

    def read(data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            # next(fp)  # 没有header，不用Skip header
            for line in fp.readlines():
                ids, words = line.strip('\n').split('\001')
                # 要预测的数据集没有label，伪造个O，不知道可以不 ，应该后面预测不会用label
                # labels = ['O' for x in range(0, len(words))]
                words_array = []
                for c in words:
                    words_array.append(c)
                yield words_array

    # 根据datafiles的数据类型，选择合适的处理方式
    if isinstance(datafiles, str):  # 字符串，单个文件名称
        # 返回单个文件对应的单个数据集
        return list(read(datafiles))
    elif isinstance(datafiles, list) or isinstance(datafiles, tuple):  # 列表或元组，多个文件名称、
        # 返回多个文件对应的多个数据集
        return [list(read(datafile)) for datafile in datafiles]



#加载测试文件
test_ds = load_testdata(datafiles=('./dataset/final_test.txt'))
# for i in range(10):
#     print(test_ds[i])
#预处理编码
for index, test in enumerate(test_ds):
    test_ds[index] = trans_func_test(test)



testloader = DataLoader(test_ds, batch_size=32, collate_fn=create_mini_batch, drop_last=False)



#8、Bert模型推理
# 将标签编码转换为标签名称，组合成预测结果
#wgm_trans_decodes(ds, pred_list, len_list, label_vocab)
# ds：Bert模型生成的编码序列列表，decodes：待转换的标签编码列表，lens：句子有效长度列表，label_vocab：标签类型KV表
def wgm_trans_decodes(ds, decodes, lens, label_vocab):
    # 将decodes和lens由列表转换为数组
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    # 先使用zip形成元祖（编号, 标签），然后使用dict形成字典
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))
    # 保存所有句子解析结果的列表
    results = []
    # 初始化编号
    inNum = 1
    # 逐个处理待转换的标签编码列表
    for idx, end in enumerate(lens):
        # 句子单字构成的数组
        sent_array = tokenizer.convert_ids_to_tokens(ds[idx][0][:end])

        # 句子单字标签构成的数组
        tags_array = [id_label[x] for x in decodes[idx][1:end]]
        # 初始化句子和解析结果
        sent = ""
        tags = ""
        # 将字符串数组转换为单个字符串
        for i in range(end - 2):
            # pdb.set_trace()
            # 单字直接连接，形成句子
            sent = sent + "|"+sent_array[i]
            # 标签以空格连接
            if i > 0:
                tags = tags + " " + tags_array[i]
            else:  # 第1个标签
                tags = tags_array[i]
        # 构成结果串：编号+句子+标签序列，中间用“\u0001”连接
        current_pred = str(inNum) + '\u0001' + sent + '\u0001' + tags + "\n"
        # pdb.set_trace()
        # 添加到句子解析结果的列表
        results.append(current_pred)
        inNum = inNum + 1
    return results


# 从标签编码中提取出地址元素
# ds：ERNIE模型生成的编码序列列表，decodes：待转换的标签编码列表，lens：句子有效长度列表，label_vocab：标签类型KV表
# 从标签编码中提取出地址元素
#elemlist = wgm_parse_decodes(ds, pred_list, len_list, label_vocab)
def wgm_parse_decodes(ds, decodes, lens, label_vocab):
    # 将decodes和lens由列表转换为数组
    decodes = [x for batch in decodes for x in batch]
    lens = [x for batch in lens for x in batch]
    # 先使用zip形成元祖（编号, 标签），然后使用dict形成字典
    id_label = dict(zip(label_vocab.values(), label_vocab.keys()))

    # 地址元素提取结果，每行是单个句子的地址元素列表
    # 例如：('朝阳区', 'district') ('小关北里', 'poi') ('000-0号', 'houseno')
    outputs = []
    for idx, end in enumerate(lens):
        # 句子单字构成的数组
        sent = tokenizer.convert_ids_to_tokens(ds[idx][0][:end])
        # 句子单字标签构成的数组
        tags = [id_label[x] for x in decodes[idx][1:end]]
        # 初始化地址元素名称和标签列表
        sent_out = []
        tags_out = []
        # 当前解析出来的地址元素名称
        words = ""
        # pdb.set_trace()
        # 逐个处理（单字, 标签）
        # 提取原理：如果当前标签是O，或者以B开头，那么说明遇到新的地址元素，需要存储已经解析出来的地址元素名称words
        # 然后，根据情况进行处理
        for s, t in zip(sent, tags):
            if t.startswith('B-') or t == 'O':  # 遇到新的地址元素
                if len(words):  # words非空，需要存储到sent_out
                    sent_out.append(words)
                if t == 'O':  # 标签为O，则直接存储标签
                    # pdb.set_trace()
                    tags_out.append(t)
                else:  # 提取出标签
                    tags_out.append(t.split('-')[1])
                # 新地址元素名称首字符
                words = s
            else:  # 完善地址元素名称
                words += s
        # 处理地址串第1个地址元素时，sent_out长度为0，和tags_out的长度不同，需要补齐
        if len(sent_out) < len(tags_out):
            sent_out.append(words)
        # 按照（名称,标签）的形式组织地址元素，并且用空格分隔开
        outputs.append(' '.join(
            [str((s, t)) for s, t in zip(sent_out, tags_out)]))
        # 换行符号
        outputs.append('\n')
    return outputs


# 使用Bert模型推理，并保存预测结果
# data_loader：
def wgm_predict_save(model, data_loader, ds, label_vocab, tagged_filename, element_filename):
    pred_list = []
    len_list = []
    for input_ids, seg_ids, lens, labels in data_loader:
        # pdb.set_trace()
        logits = model(input_ids=input_ids, attention_mask=seg_ids)
        pred = torch.argmax(logits[0], axis=-1)
        # print(pred)
        pred_list.append(pred.numpy())
        len_list.append([len.item() for len in lens])
    # 将标签编码转换为标签名称，组合成预测结果
    predlist = wgm_trans_decodes(ds, pred_list, len_list, label_vocab)
    # 从标签编码中提取出地址元素
    elemlist = wgm_parse_decodes(ds, pred_list, len_list, label_vocab)
    # 保存预测结果
    with open(tagged_filename, 'w', encoding='utf-8') as f:
        f.writelines(predlist)
    # 保存地址元素
    with open(element_filename, 'w', encoding='utf-8') as f:
        f.writelines(elemlist)



#加载Bert模型
# a.通过词典导入分词器
# tokenizer = BertTokenizer.from_pretrained("./dataset/model/bert-base-chinese/")
# b. 导入配置文件
# model_config = BertConfig.from_pretrained("./dataset/model/bert-base-chinese/")
model = BertForTokenClassification.from_pretrained("./bert_result/", num_labels=len(label_vocab))
# bert_model = BertModel.from_pretrained(MODEL_PATH, config=model_config)
# model=BertTokenizer.from_pretrained("./bert_result/")
# model_dict = torch.load('./bert_result/')
# model.set_dict(model_dict)

#推理并预测结果
wgm_predict_save(model, testloader, test_ds, label_vocab, "predict_wgm.txt", "element_wgm.txt")

