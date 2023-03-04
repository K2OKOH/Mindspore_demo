#!/usr/bin/env python
# coding: utf-8

from DataSet import MyDatasets
import mindspore.nn as nn

from models import *
from cell import *
import sys
import os
import numpy as np
from test_PACS import test
import tqdm as tqdm
from PIL import Image
import cv2

# ========================================================

learning_rate=0.0001

print('dataset loading')
Train_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = '../DataSet/PACS/photo'), column_names=["image", "label"])
Train_DataSet = Train_DataSet.batch(256)
print('load finished!')

def train():
    net = AlexNet(num_classes=7)

    opt_net = nn.Adam(net.get_parameters(), learning_rate=learning_rate)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = CE_net(net, criterion)
    train_network = TrainOneStepCell_PACS(net_with_criterion, opt_net)
    train_network.set_train()

    for epoch in range(60):
        num = 0
        for data in Train_DataSet.create_dict_iterator():
            img_s = data["image"]
            label_s = data["label"]
            '''
            for i in range(256):
                print(i)
                # print(img_s.shape)
                img = img_s[i].asnumpy().transpose((1, 2, 0))
                # print(img.shape)
                # img = Image.fromarray(img, mode='RGB')

                # # 将PIL Image对象转换为NumPy数组
                # img_np = np.array(img)

                # 保存图片
                dir_path = './' + str(int(label_s[i]))
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                
                cv2.imwrite(dir_path + '/my_image{}.jpg'.format(i), img)

            # sys.exit(0)
            '''
            loss = train_network(img_s, label_s)
            
        print('Epoch:',epoch,'loss = ', loss.asnumpy())
            # 测试数据迭代
        test_number = 0
        correct_number = 0
        all_picture = 0

        for data in Train_DataSet.create_dict_iterator():
            test_number += 1
            
            img_T = data["image"]
            label_T = data["label"]

            onehot_label = net(img_T)

            pre_label = np.argmax(onehot_label, axis=1)
            # print("pre_label:", pre_label)
            # print("label_T:", label_T)
            
            correct_number += sum(pre_label == label_T)

            all_picture += len(pre_label)
        
            # if (test_number%100 == 0):
            #     print('已测试：\t', test_number)
            
        acc  = correct_number / all_picture

        print("Accuracy is {:.2f}% {}/{}\n".format(acc, correct_number, all_picture))
        # 保存模型
        save_model(net, './SaveModel/model_PACS_P.ckpt')
        print('>> PACS_P 保存完成')

        # test()

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