#!/usr/bin/env python
# coding: utf-8

# from mindvision.dataset import Mnist
from DataSet import MyDatasets
import mindspore.nn as nn

from models import *
from cell import *
import sys
import numpy as np
from test_PACS_all_resnet import test
import tqdm as tqdm
from mindspore import context

# ========================================================

learning_rate=0.0001

print('dataset loading')
Photo_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
Photo_DataSet_s2 = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo_s2'), column_names=["image", "label"])
data_train = Photo_DataSet.batch(32)
data_train_s2 = Photo_DataSet_s2.batch(32)
print('load finished!')

def train():
    net = Resnet_MV3(num_classes=7)

    opt_net = nn.Adam(net.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    ce_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    mse_loss = nn.MSELoss(reduction='mean')
    net_with_criterion = MV3_loss_net(net, ce_loss, mse_loss)
    train_network = TrainOneStepCell_MV3(net_with_criterion, opt_net)
    train_network.set_train()

    for epoch in range(60):
        num = 0
        loss_list = []
        for data in data_train.create_dict_iterator():
            num += 1
            img_s1 = data["image"]
            label_s1 = data["label"]
            data_s2 = next(iter(data_train_s2.create_dict_iterator()))
            img_s2 = data_s2["image"]
            label_s2 = data_s2["label"]

            loss = train_network(img_s1, img_s2, label_s1, label_s2)
            loss_list.append(loss.asnumpy())
            # if (num%10 == 0):
            #     print(num)
            
        mean_loss = sum(loss_list)/len(loss_list)
        print('Epoch:',epoch,'loss = ', mean_loss)
        
        # 保存模型
        save_model(net, './SaveModel/res_MV3_P.ckpt')
        print('>> PACS_P 保存完成')
            
        test()

# 保存模型
def save_model(model, name):
    mindspore.save_checkpoint(model, name)

if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)
    print('>> 开始训练')
    train()
    print('>> 完成训练\n')
    
    # 进行测试
    print('>> 开始测试')
    test()
    print('>> 完成测试')