#!/usr/bin/env python
# coding: utf-8

# from mindvision.dataset import Mnist
from DataSet import MyDatasets
import mindspore.nn as nn

from models import *
from cell import *
import sys
import numpy as np
from test_PACS import test
import tqdm as tqdm

# ========================================================

learning_rate=0.0002

print('dataset loading')
Photo_DataSet_s1 = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
data_train_s1 = Photo_DataSet_s1.batch(257)
Photo_DataSet_s2 = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
data_train_s2 = Photo_DataSet_s2.batch(256)
print('load finished!')

def train():
    net = AlexNet(num_classes=7)

    opt_net = nn.Adam(net.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    train_network = TrainOneStepCell_MLRT(net, criterion, opt_net)
    train_network.set_train()

    for epoch in range(20):
        num = 0
        for data in data_train_s1.create_dict_iterator():
            # num += 1
            img_s1 = data["image"]
            label_s1 = data["label"]
            data_s2 = next(iter(data_train_s2.create_dict_iterator()))
            img_s2 = data_s2["image"]
            label_s2 = data_s2["label"]

            loss = train_network(img_s1, img_s2, label_s1, label_s2)

            # if(num%10 == 0):
            #     print(num)
            
        print('Epoch:',epoch,'loss = ', loss.asnumpy())
        
        # 保存模型
        save_model(net, './SaveModel/model_MLRT_P.ckpt')
        print('>> MLRT_P 保存完成')

        test()

# 保存模型
def save_model(model, name):
    mindspore.save_checkpoint(model, name)

if __name__ == "__main__":
    print('>> 开始训练')
    train()
    print('>> 完成训练\n')
    
    # 进行测试
    print('>> 开始测试')
    test()
    print('>> 完成测试')