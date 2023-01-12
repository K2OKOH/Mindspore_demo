#!/usr/bin/env python
# coding: utf-8

from mindvision.dataset import Mnist
import mindspore.nn as nn

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
    mnist_test = Mnist(path="./mnist", split="test", batch_size=32, repeat_num=1, shuffle=True, resize=28)
    test_dataset = mnist_test.run()
    print('>> load finished!')

    # 读取模型
    Gen4_T = Generator()
    Cls_T = Classifier()
    Cls1_T = Classifier()
    Cls2_T = Classifier()

    Gen4_dict = mindspore.load_checkpoint('./SaveModel/model_Gen4.ckpt')
    param_not_load = mindspore.load_param_into_net(Gen4_T, Gen4_dict)
    Cls_dict = mindspore.load_checkpoint('./SaveModel/model_Cls.ckpt')
    param_not_load = mindspore.load_param_into_net(Cls_T, Cls_dict)
    Cls1_dict = mindspore.load_checkpoint('./SaveModel/model_Cls1.ckpt')
    param_not_load = mindspore.load_param_into_net(Cls1_T, Cls_dict)
    Cls2_dict = mindspore.load_checkpoint('./SaveModel/model_Cls2.ckpt')
    param_not_load = mindspore.load_param_into_net(Cls2_T, Cls_dict)
    if (not param_not_load):
        print('模型参数读取成功')
    else:
        print('模型读取失败')
    
    
    # 测试数据迭代
    test_number = 0
    correct1_number = 0
    correct21_number = 0
    correct22_number = 0
    correcte1_number = 0
    correcte2_number = 0
    all_picture = 0

    for data in test_dataset.create_dict_iterator():
        test_number += 1
        
        img_T = data["image"]
        label_T = data["label"]

        feat = Gen4_T(img_T)
        onehot1_label = Cls_T(feat)
        onehot21_label = Cls1_T(feat)
        onehot22_label = Cls2_T(feat)

        ensemble_cc1 = onehot1_label + onehot21_label
        ensemble_cc1c2 = onehot1_label + onehot21_label + onehot22_label

        pre1_label = np.argmax(onehot1_label, axis=1)
        pre21_label = np.argmax(onehot21_label, axis=1)
        pre22_label = np.argmax(onehot22_label, axis=1)
        pree1_label = np.argmax(ensemble_cc1, axis=1)
        pree2_label = np.argmax(ensemble_cc1c2, axis=1)
        
        correct1_number += sum(pre1_label == label_T)
        correct21_number += sum(pre21_label == label_T)
        correct22_number += sum(pre22_label == label_T)
        correcte1_number += sum(pree1_label == label_T)
        correcte2_number += sum(pree2_label == label_T)

        all_picture += len(pre1_label)
    
        if (test_number%100 == 0):
            print('已测试：\t', test_number)
        
    acc1  = correct1_number / all_picture
    acc21 = correct21_number / all_picture
    acc22 = correct22_number / all_picture
    acce1 = correcte1_number / all_picture
    acce2 = correcte2_number / all_picture

    acc = max(acc1, acc21, acc22, acce1, acce2)
    
    print("Accuracy is {:.2f}%".format(acc))

if __name__ == "__main__":
    # 进行测试
    print('>>start test')
    test()
    print('>>测试完成')