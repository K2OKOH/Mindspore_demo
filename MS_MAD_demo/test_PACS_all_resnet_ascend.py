#!/usr/bin/env python
# coding: utf-8

# from mindvision.dataset import Mnist
import mindspore.nn as nn
from DataSet import MyDatasets

from models import *
from cell import *
import sys
import numpy as np
from tqdm import tqdm
from resnet import resnet50
import argparse

# ========================================================
parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
parser.add_argument('--dataset_path', dest='dataset_path', default='obs://xmj/PACS/PACS', type=str)
parser.add_argument('--model_path', dest='model_path', default='obs://xmj/PACS/MAD_IC/SaveModel/res_MV3_P.ckpt', type=str)
args = parser.parse_args()

max_acc = [0,0,0,0]

def test():
    # 读取数据集
    print('Test dataset loading ...')
    Photo_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path = args.dataset_path + 'photo'), column_names=["image", "label"])
    Photo_DataSet = Photo_DataSet.batch(128)
    Art_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path =  args.dataset_path + 'art_painting'), column_names=["image", "label"])
    Art_DataSet = Art_DataSet.batch(128)
    Cartoon_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path =  args.dataset_path + 'cartoon'), column_names=["image", "label"])
    Cartoon_DataSet = Cartoon_DataSet.batch(128)
    Sketch_DataSet = mindspore.dataset.GeneratorDataset(MyDatasets(data_path =  args.dataset_path + 'sketch'), column_names=["image", "label"])
    Sketch_DataSet = Sketch_DataSet.batch(128)
    print('>> load finished!')

    # 读取模型
    net_test = Resnet(num_classes=7)

    net_dict = mindspore.load_checkpoint(args.model_path)
    param_not_load = mindspore.load_param_into_net(net_test, net_dict)
    print('Model loading...')
    
    # 测试数据迭代
    # P
    test_number = 0
    correct_number = 0
    all_picture = 0
    for data in Photo_DataSet.create_dict_iterator():
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

    print("DataSet P->P Accuracy is {:.2f}% {}/{}".format(acc*100, correct_number, all_picture))
    if  (acc*100 > max_acc[0]):
        max_acc[0] = acc*100
    
    # A
    test_number = 0
    correct_number = 0
    all_picture = 0
    for data in Art_DataSet.create_dict_iterator():
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
    print("DataSet P->A Accuracy is {:.2f}% {}/{}".format(acc*100, correct_number, all_picture))
    if  (acc*100 > max_acc[1]):
        max_acc[1] = acc*100

    # C
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
    print("DataSet P->C Accuracy is {:.2f}% {}/{}".format(acc*100, correct_number, all_picture))
    if  (acc*100 > max_acc[2]):
        max_acc[2] = acc*100
    
    # S
    test_number = 0
    correct_number = 0
    all_picture = 0
    for data in Sketch_DataSet.create_dict_iterator():
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
    print("DataSet P->S Accuracy is {:.2f}% {}/{}".format(acc*100, correct_number, all_picture))
    if  (acc*100 > max_acc[3]):
        max_acc[3] = acc*100

    print("P : ", max_acc[0])
    print("A : ", max_acc[1])
    print("C : ", max_acc[2])
    print("S : ", max_acc[3])

if __name__ == "__main__":
    # 进行测试
    print('>>start test')
    test()
    print('>>测试完成')