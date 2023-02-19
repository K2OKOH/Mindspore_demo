#!/usr/bin/env python
# coding: utf-8

# from mindvision.dataset import Mnist
import mindspore.nn as nn
from DataSet import MyDatasets

from models import *
from cell import *
import sys
import numpy as np

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

def test():
    # 读取数据集
    print('Test dataset loading ...')
    Cartoon_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
    Cartoon_DataSet = Cartoon_DataSet.batch(256)
    print('>> load finished!')

    # 读取模型
    net_test = AlexNet(num_classes=7)

    net_dict = mindspore.load_checkpoint('./SaveModel/model_MV3_P.ckpt')
    param_not_load = mindspore.load_param_into_net(net_test, net_dict)
    # if (not param_not_load):
    #     print('模型参数读取成功')
    # else:
    #     print('模型读取失败')
    
    
    # 测试数据迭代
    test_number = 0
    correct_number = 0
    all_picture = 0

    for data in Cartoon_DataSet.create_dict_iterator():
        test_number += 1
        
        img_T = data["image"]
        label_T = data["label"]

        onehot_label = net_test(img_T)

        pre_label = np.argmax(onehot_label, axis=1)
        
        correct_number += sum(pre_label == label_T)

        all_picture += len(pre_label)
    
        # if (test_number%100 == 0):
        #     print('已测试：\t', test_number)
        
    acc  = correct_number / all_picture

    print("Accuracy is {:.2f}%".format(acc))

if __name__ == "__main__":
    # 进行测试
    print('>>start test')
    test()
    print('>>测试完成')