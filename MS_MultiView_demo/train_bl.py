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

batch_size=32
source='usps'
target='mnist'
learning_rate=0.0002
interval=100
optimizer='adam'
num_k=4, 
all_use=False
checkpoint_dir=None
save_epoch=10

class_num = 10
num_k1 = 8
num_k2 = 1
num_k3 = 8
num_k4 = 1

print('dataset loading')
Photo_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
data_train = Photo_DataSet.batch(256)
print('load finished!')

def train():
    net = AlexNet(num_classes=7)

    opt_net = nn.Adam(net.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = CE_net(net, criterion)
    train_network = TrainOneStepCell_PACS(net_with_criterion, opt_net)
    train_network.set_train()

    for epoch in range(20):
        num = 0
        for data in data_train.create_dict_iterator():
            img_s = data["image"]
            label_s = data["label"]

            loss = train_network(img_s, label_s)
            
        print('Epoch:',epoch,'loss = ', loss.asnumpy())
        
        # 保存模型
        save_model(net, './SaveModel/model_PACS_P.ckpt')
        print('>> PACS_P 保存完成')

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