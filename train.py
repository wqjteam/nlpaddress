import torch
import numpy as np
import os
def format_data(source_filename, target_filename):
    datalist = []
    # 读取source_filename所有数据到lines中，每个元素是字标注
    with open(source_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # 逐个处理每个字标注
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