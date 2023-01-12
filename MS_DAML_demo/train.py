#!/usr/bin/env python
# coding: utf-8

from mindvision.dataset import Mnist
import mindspore.nn as nn

from models import *
from cell import *
import sys
import numpy as np
from test import test

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
mnist = Mnist(path="./mnist", split="train", batch_size=32, repeat_num=1, shuffle=True, resize=28)
dataset_infer = mnist.run()
print('load finished!')

mnist_test = Mnist(path="./mnist", split="test", batch_size=32, repeat_num=1, shuffle=True, resize=28)
dataset_test = mnist_test.run()

def train():

    Gen1 = Generator()
    Gen2 = Generator()
    Gen3 = Generator()
    Gen4 = Generator()
    Cls = Classifier()
    Cls1 = Classifier()
    Cls2 = Classifier()
    Dis = Discriminator()
    Mix = Mixer()

    Gen14_C = G14_C_net(Gen1, Gen4, Cls)
    Gen1_D = G1_D_net(Gen1, Dis)
    Gen12 = G12_net(Gen1, Gen2)
    Gen3_C = G3_C_net(Gen3, Cls)
    Gen4_Cs = G4_Cs_net(Gen4, Cls, Cls1, Cls2)
    Gen1234_M = G1234_M_net(Gen1, Gen1, Gen1, Gen1, Mix)
    Ite4 = Ite4_net(Gen1, Gen2, Gen3, Gen4, Cls, Cls1, Cls2, Dis, Mix)

    # Lce
    opt_g1_c = nn.Adam(Gen14_C.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = CE_net(Gen14_C, criterion)
    train_network = TrainOneStepCell(net_with_criterion, opt_g1_c)
    train_network.set_train()
    # di
    opt_g1_d = nn.Adam(Gen1_D.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    BCELoss = nn.BCEWithLogitsLoss()
    G1_with_BCE = G1_net_loss(Gen1_D, BCELoss)
    G1_with_BCE_network = TrainOneStepCell_G1_BCE(G1_with_BCE, opt_g1_d)
    G1_with_BCE_network.set_train()
    # ds
    opt_g12 = nn.Adam(Gen12.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    MSLoss = MeanSoftmax()
    G12_with_MS = G12_net_loss(Gen12, MSLoss)
    G12_with_MS_network = TrainOneStepCell_G12_MS(G12_with_MS, opt_g12)
    G12_with_MS_network.set_train()
    # ds
    opt_g3_c = nn.Adam(Gen3_C.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    G3_C_with_Loss = G3_C_net_loss(Gen3_C)
    G3_C_with_Loss_network = TrainOneStepCell_G3_C_Loss(G3_C_with_Loss, opt_g3_c)
    G3_C_with_Loss_network.set_train()
    # cs
    opt_g4_cs = nn.Adam(Gen4_Cs.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    G4_Cs_with_Loss = G4_Cs_net_loss(Gen4_Cs, criterion)
    G4_Cs_with_Loss_network = TrainOneStepCell_G4_Cs_Loss(G4_Cs_with_Loss, opt_g4_cs)
    G4_Cs_with_Loss_network.set_train()
    # m
    opt_g1234_m = nn.Adam(Gen1234_M.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    G1234_M_with_Loss = G1234_M_net_loss(Gen1234_M)
    G1234_M_with_Loss_network = TrainOneStepCell_G1234_M_Loss(G1234_M_with_Loss, opt_g1234_m)
    G1234_M_with_Loss_network.set_train()
    # I4
    opt_i4 = nn.Adam(Ite4.get_parameters(), learning_rate=learning_rate, weight_decay=0.0005)
    I4_with_Loss = I4_net_loss(Ite4, BCELoss)
    I4_with_Loss_network = TrainOneStepCell_I4_Loss(I4_with_Loss, opt_i4)
    I4_with_Loss_network.set_train()

    for epoch in range(10):
        num = 0
        for data in dataset_infer.create_dict_iterator():
            img_s = data["image"]
            img_t = data["image"]
            label_s = data["label"]

            # source domain is discriminative.
            for i1 in range(num_k1):
                # Lce
                loss_G12_C = train_network(img_s, label_s)
            
            # transferability (di)
            for i2 in range(num_k2):
                loss_G1_D = G1_with_BCE_network(img_s, img_t)
                
                # transferability (ds)
                loss_G12 = G12_with_MS_network(img_s, img_t)

                ####### discriminablity(ci) 
                loss_G3_C = G3_C_with_Loss_network(img_t)

                # discriminablity(cs)
                loss_G4_Cs = G4_Cs_with_Loss_network(img_s, img_t, label_s)

            for i3 in range(num_k3):
                # continue
                loss_G1234_M = G1234_M_with_Loss_network(img_t)

            for i4 in range(num_k4):
                loss_i4 = I4_with_Loss_network(img_s, img_t)

            loss = loss_G12_C

            num += 1
            if (num%100 == 0):
                print('num:', num, 'loss:', loss.asnumpy())

        print('Epoch:',epoch,'loss = ', loss.asnumpy())
        
        # 保存模型
        save_model(Gen4, './SaveModel/model_Gen4.ckpt')
        print('>> Gen4 保存完成')
        save_model(Cls, './SaveModel/model_Cls.ckpt')
        print('>> Cls 保存完成')
        save_model(Cls1, './SaveModel/model_Cls1.ckpt')
        print('>> Cls1 保存完成')
        save_model(Cls2, './SaveModel/model_Cls2.ckpt')
        print('>> Cls2 保存完成')

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