#import torch
#import torch.nn.functional as F
#from torch.autograd import Variable
#import torch.autograd as autograd
#import torch.optim as optim
#import torch.nn.init as init
#import torch.nn as nn
import mindspore
import mindspore.nn as nn
from mindspore import set_seed
from mindspore import Tensor
import mindspore.context as context
from mindspore import ops
from mindspore.ops import ExpandDims, Argmax
import mindspore.numpy as mnp
from mindspore import Model
from mindspore.common import initializer as init
from mindspore import save_checkpoint
from cell import APnetWithLossCell, RnetWithLossCell, ReconnetWithLossCell, TrainOneStepCell
from mindspore.common.initializer import initializer, XavierUniform, Zero

from time import gmtime, strftime
import numpy as np
import argparse
import os
import glob
import random
import scipy.io as sio

from dataset_loader import FeatDataLayer, DATA_LOADER
from models import _AttributeNet, _RelationNet, _param

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--dataroot', default='/media/cqu/D/SWL/TCN/datasets/data_resnet2',
                    help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--exp_idx', default='', type=str, help='exp idx')
#parser.add_argument('--manualSeed', type=int, default=4083, help='manual seed')
parser.add_argument('--manualSeed', type=int, default=4283, help='manual seed')
parser.add_argument('--resume', type=str, help='the model to resume')
parser.add_argument('--z_dim', type=int, default=100, help='dimension of the random vector z')
parser.add_argument('--disp_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval', type=int, default=200)
parser.add_argument('--beta', type=int, default=0.01)
parser.add_argument('--lambda_1',   type=float,  default=0.1)
parser.add_argument('--lambda_2',   type=float,  default=0.1)
parser.add_argument('--lambda_3',   type=float,  default=0.001)

opt = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

""" hyper-parameter  """
opt.REG_W_LAMBDA = 0.01

opt.lr = 0.0001
opt.batch_size = 64  # 512

""" hyper-parameter for testing"""

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
set_seed(opt.manualSeed)
#torch.manual_seed(opt.manualSeed)
#torch.cuda.manual_seed_all(opt.manualSeed)

class Recon(nn.Cell):
    def __init__(self, x_dim, y_dim):
        super(Recon, self).__init__()
        z_dim = 1024
        self.fc = nn.Dense(x_dim, z_dim)
        self.fc1 = nn.Dense(z_dim, y_dim)
        self.relu = nn.ReLU()
        self.Dropout = nn.Dropout(keep_prob=0.5)
        self.Sigmoid = nn.Sigmoid()

    def construct(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.Dropout(x)

        x = self.fc1(x)
        x = self.Sigmoid(x)
        return x


def data_iterator(train_x, train_att):
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        #print(len(train_x))
        idxs = np.arange(0, len(train_x))
        np.random.shuffle(idxs)
        shuf_visual = train_x[idxs]
        shuf_att = train_att[idxs]
        batch_size = 64


        for batch_idx in range(0, len(train_x), batch_size):
            #print(batch_idx)
            visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
            visual_batch = visual_batch.astype("float32")
            label_batch = shuf_att[batch_idx:batch_idx + batch_size]

            #label_batch = torch.from_numpy(label_batch).float()
            #visual_batch = torch.from_numpy(visual_batch).float()
            label_batch = Tensor(label_batch.astype(np.int32)) 
            visual_batch = Tensor(visual_batch.astype(np.float32))
            yield label_batch, visual_batch


all_feature_set = sio.loadmat('/media/cqu/D/SWL/Feature-Generating-Networks-master/saved_models/feature_test_gan_cub.mat')
test_feature     = all_feature_set['all_feature']
#test_feature     = normalize(test_feature)
test_label       = np.squeeze(all_feature_set['all_label'].T)
iter_ = data_iterator(test_feature, test_label)


def train():
    param = _param()
    dataset = DATA_LOADER(opt)
    param.X_dim = dataset.feature_dim

    data_layer = FeatDataLayer(dataset.train_label.asnumpy(), dataset.train_feature.asnumpy(), opt)
    result = Result()
    result_gzsl = Result()


    #APnet = _AttributeNet(dataset.att_dim, dataset.feature_dim).cuda()
    APnet = _AttributeNet(dataset.att_dim, dataset.feature_dim)
    #APnet.apply(weights_init)
    weights_init_ms(APnet)
    print(APnet)

    #Rnet = _RelationNet(dataset.feature_dim).cuda()
    Rnet = _RelationNet(dataset.feature_dim)
    #Rnet.apply(weights_init)
    weights_init_ms(Rnet)
    print(Rnet)
    
    #Reconnet = Recon( dataset.feature_dim, dataset.att_dim).cuda()
    Reconnet = Recon(dataset.feature_dim, dataset.att_dim)
    #Reconnet.apply(weights_init)
    weights_init_ms(Reconnet)
    print(Reconnet)

    exp_info = 'GBU_{}'.format(opt.dataset)
    exp_params = 'Rls{}'.format(opt.REG_W_LAMBDA)

    out_dir = 'Result/{:s}'.format(exp_info)
    out_subdir = 'Result/{:s}/{:s}'.format(exp_info, exp_params)
    if not os.path.exists('Result'):
        os.mkdir('Result')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    if not os.path.exists(out_subdir):
        os.mkdir(out_subdir)
    #cprint(" The output dictionary is {}".format(out_subdir), 'red')
    print(" The output dictionary is {}".format(out_subdir))
    log_dir = out_subdir + '/log_{:s}_{}.txt'.format(exp_info, opt.exp_idx)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0

    '''
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            APnet.load_state_dict(checkpoint['state_dict_AP'])
            start_step = checkpoint['it']
            print(checkpoint['log'])
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    '''
    
    #optimizerAP = optim.Adam(APnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    #optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    #optimizerRR = optim.Adam(Reconnet.parameters(), lr=opt.lr, betas=(0.5, 0.9))
    #optimizerAP = nn.Adam(APnet.trainable_params(), learning_rate=opt.lr, betas=(0.5, 0.9))
    #optimizerR = nn.Adam(Rnet.trainable_params(), learning_rate=opt.lr, betas=(0.5, 0.9))
    #optimizerRR = nn.Adam(Reconnet.trainable_params(), learning_rate=opt.lr, betas=(0.5, 0.9))
    
    op = ops.Concat()
    all_class = op((dataset.seenclasses, dataset.unseenclasses))
    transpose = ops.Transpose()
    zeros = ops.Zeros()
    scatter = ops.ScatterNd()
    cast = ops.Cast()
    type_dst = mindspore.float32
    
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss(reduction='mean')
    sigmoid = nn.Sigmoid()
    
    APnet_with_loss = APnetWithLossCell(APnet, mse)
    Rnet_with_loss  = RnetWithLossCell(APnet, Rnet, bce, all_class.shape[0], dataset.train_class.asnumpy().shape[0], opt.lambda_3)
    Reconnet_with_loss = ReconnetWithLossCell(APnet, Reconnet, mse)
    
    optimizerAP = nn.Adam(APnet.trainable_params(), learning_rate=opt.lr, beta1=0.5, beta2=0.9)
    optimizerR = nn.Adam(Rnet.trainable_params(), learning_rate=opt.lr, beta1=0.5, beta2=0.9)
    optimizerRR = nn.Adam(Reconnet.trainable_params(), learning_rate=opt.lr, beta1=0.5, beta2=0.9)
    
    net_train = TrainOneStepCell(APnet_with_loss, Rnet_with_loss, Reconnet_with_loss,
                                 optimizerAP, optimizerR, optimizerRR)
    
    
    #nets = [APnet, Rnet, Reconnet]

    APnet.set_train()
    Rnet.set_train()
    Reconnet.set_train()

    acc = 0
    H = 0
    S = 0
    U = 0
    
    best_zsl=0
    best_H  =0
    best_S  =0
    best_U  =0

    #all_class = torch.cat([dataset.seenclasses, dataset.unseenclasses], 0)
    

    for it in range(start_step, 50000 + 1):
        blobs = data_layer.forward()
        batch_feats = blobs['data']  # image data
        batch_labels = blobs['labels']  # class labels
        #batch_atts  =  dataset.attribute[batch_labels].cuda()
        #support_attr = dataset.attribute[all_class].cuda()
        batch_labels = Tensor(batch_labels)
        batch_atts  =  dataset.attribute[batch_labels]
        support_attr = dataset.attribute[all_class]
        class_num = all_class.shape[0]
        
        batch_labels = batch_labels.asnumpy()
        batch_images = Tensor(batch_feats)
        

        unsqu = ExpandDims()
        batch_ext = unsqu(batch_images, 0)
        batch_ext = mnp.tile(batch_ext, (class_num, 1, 1))
        input_perm = (1, 0, 2)
        batch_ext = transpose(batch_ext, input_perm)
        
        
        #unseen samples
        label_batch_val, visual_batch_val = next(iter_)
        att_batch_val   = dataset.attribute[label_batch_val]
        
        # forward
        '''
        semantic_proto = APnet(support_attr)   #50x2048
        semantic_proto_batch = APnet(batch_atts)
        semantic_proto_ext = unsqu(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (opt.batch_size, 1, 1))


        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        unseen_semantic_proto_batch = APnet(att_batch_val)

        rec_sem           =  Reconnet(semantic_proto_batch)
        rec_sem_unseen    =  Reconnet(unseen_semantic_proto_batch)
        '''
        
        # label transform
        support_labels = dataset.train_class.asnumpy()
        re_batch_labels = np.zeros_like(batch_labels)
        train_class_num = support_labels.shape[0]
        

        for i in range(train_class_num):
            re_batch_labels[batch_labels == support_labels[i]] = i
        #print(re_batch_labels)
        #re_batch_labels = torch.LongTensor(re_batch_labels)
        re_batch_labels = Tensor(re_batch_labels.astype(np.int32))

        #one_hot_labels = torch.zeros(opt.batch_size, train_class_num).scatter_(1, re_batch_labels.view(-1, 1), 1).cuda()
        #one_hot_labels = zeros((opt.batch_size, train_class_num), mindspore.int32)
        updates = Tensor(np.ones(opt.batch_size), mindspore.int32)
        #print(updates.shape)
        m1 = np.arange(opt.batch_size)
        m1.reshape((opt.batch_size, 1))
        #print(m1.shape)
        #print(re_batch_labels.view(-1, 1).shape)
        m1 = Tensor(m1.astype(np.int32))
        concat = ops.Concat(1)
        indices = concat((m1.view(-1, 1), re_batch_labels.view(-1, 1)))
        one_hot_labels = scatter(indices, updates, (opt.batch_size, train_class_num))
        one_hot_labels = cast(one_hot_labels, type_dst)
        
        #print(one_hot_labels.shape)
        #print(one_hot_labels[0])
        #sim_labels =  dataset.sim_re_2[re_batch_labels].cuda()
        sim_labels =  dataset.sim_re_2[re_batch_labels]

        '''
        relation_train = relations[:, :train_class_num]
        relation_test = relations[:, train_class_num:]
        '''
        #print('0430')
        #print(batch_images.shape)
        loss, out1, out2, out3 = net_train(opt.lambda_1, opt.lambda_2, opt.lambda_3, batch_images, batch_atts, visual_batch_val, att_batch_val, one_hot_labels, sim_labels, support_attr, batch_ext)
        
        

        #训练类别损失
        '''
        loss1 = bce(relation_train, one_hot_labels)
        loss2 = bce(relation_test, sim_labels)
        loss3 = mse(sigmoid(semantic_proto_batch), sigmoid(batch_images))
        loss4 = mse(sigmoid(rec_sem), sigmoid(batch_atts))  #训练样本重构误差
        loss5 = mse(sigmoid(unseen_semantic_proto_batch), sigmoid(visual_batch_val))  #训练样本重构误差
        loss7 = mse(sigmoid(rec_sem_unseen), sigmoid(att_batch_val))  #训练样本重构误差
        loss = loss1 + opt.lambda_3* loss2+ opt.lambda_1*loss3+  opt.lambda_1*loss5 + opt.lambda_2*loss4 + opt.lambda_2*loss7
        '''

        if it % opt.evl_interval == 0 and it >= 100:
            #APnet.set_eval()
            #Rnet.set_eval()
            acc = eval_test(it, APnet, Rnet, dataset, param, result)
            if result.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_ZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                # best_acc = result.acc_list[-1]
                save_model(it, APnet, Rnet,Reconnet, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_ZSL_Acc_{:.2f}.tar'.format(result.acc_list[-1]))

            H, S, U = eval_test_gzsl(it, APnet, Rnet, dataset, param, result_gzsl)
            
            if acc>best_zsl:
                best_zsl = acc
            if H>best_H:
                best_H= H
                best_S= S
                best_U= U

            print("best_S{:.2f}%  best_U {:.2f} best_H {:.1f}%  best_zsl {:.2f}%   ".format(best_S, best_U, best_H, best_zsl))
            
            
            if result_gzsl.save_model:
                files2remove = glob.glob(out_subdir + '/Best_model_GZSL_*')
                for _i in files2remove:
                    os.remove(_i)
                # best_acc_gzsl = result.acc_list[-1]
                save_model(it, APnet, Rnet, Reconnet, opt.manualSeed, log_text,
                           out_subdir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl.best_acc,
                                                                                                 result_gzsl.best_acc_S_T,
                                                                                                 result_gzsl.best_acc_U_T))

            #APnet.train()
            #Rnet.train()
            APnet.set_train()
            Rnet.set_train()

        if it % opt.save_interval == 0 and it:
            save_model(it, APnet, Rnet, Reconnet, opt.manualSeed, log_text,
                       out_subdir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_subdir + '/Iter_{:d}.tar'.format(it))

        if it % opt.disp_interval == 0 and it:
            
            log_text = 'Iter-{}; Loss: {:.3f}; ZSL: {:.3f}; H: {:.3f}; S: {:.3f}; U: {:.3f};'.format(it, loss.asnumpy() , acc, H, S, U)
            print(log_text)
            with open(log_dir, 'a') as f:
                f.write(log_text + '\n')

'''
def save_model(it, APnet, Rnet,Reconnet, random_seed, log, fout):
    torch.save({
        'it': it + 1,
        'state_dict_AP': APnet.state_dict(),
        'state_dict_R':  Rnet.state_dict(),
        'state_dict_Rec': Reconnet.state_dict(),
        'random_seed': random_seed,
        'log': log,
    }, fout)
'''

def save_model(it, APnet, Rnet,Reconnet, random_seed, log, fout):
    save_checkpoint(APnet, "APnet.ckpt")

def eval_test(it, APnet, Rnet, dataset, param, result):
    transpose = ops.Transpose()
    #support_attr = torch.from_numpy(dataset.test_att).cuda()
    support_attr = Tensor(dataset.test_att)
    class_num = support_attr.shape[0]

    # label transform
    support_labels = np.array(dataset.unseenclasses)
    test_labels = np.array(dataset.test_unseen_label)

    #test_images = dataset.test_unseen_feature.cuda()
    test_images = dataset.test_unseen_feature
    preds = np.zeros_like(test_labels)

    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        #batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        #batch_ext = torch.transpose(batch_ext, 0, 1)
        unsqu1 = ExpandDims()
        batch_ext = unsqu1(batch_images, 0)
        batch_ext = mnp.tile(batch_ext, (class_num, 1, 1))
        input_perm = (1, 0, 2)
        batch_ext = transpose(batch_ext, input_perm)
        
        

        # forward
        semantic_proto = APnet(support_attr)
        #semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        semantic_proto_ext = unsqu1(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (end - start, 1, 1))
        
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        #prob = relations.data.cpu().numpy()
        prob = relations.asnumpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    grounds = test_labels
    preds = preds.astype(np.int32)
    predicts = support_labels[preds]

    acc = np.zeros(class_num)
    for i in range(class_num):
        index = grounds == support_labels[i]
        acc[i] = (predicts[index] == support_labels[i]).mean()
    acc = acc.mean() * 100

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.save_model = True
    print("Accuracy is {:.2f}%".format(acc))
    return acc


def eval_test_gzsl(it, APnet, Rnet, dataset, param, result):
    transpose = ops.Transpose()
    #support_attr = dataset.attribute.cuda()
    support_attr = dataset.attribute
    class_num = support_attr.shape[0]

    # label transform
    support_labels = np.array(dataset.allclasses)

    """  S -> T
    """
    test_labels = np.array(dataset.test_seen_label)
    #test_images = dataset.test_seen_feature.cuda()
    test_images = dataset.test_seen_feature
    preds = np.zeros_like(test_labels)

    unsqu1 = ExpandDims()
    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        #batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        #batch_ext = torch.transpose(batch_ext, 0, 1)
        
        batch_ext = unsqu1(batch_images, 0)
        batch_ext = mnp.tile(batch_ext, (class_num, 1, 1))
        input_perm = (1, 0, 2)
        batch_ext = transpose(batch_ext, input_perm)

        # forward
        semantic_proto = APnet(support_attr)
        #semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        semantic_proto_ext = unsqu1(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (end - start, 1, 1))
        
        
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        #prob = relations.data.cpu().numpy()
        prob = relations.asnumpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    label_T = test_labels
    num_seen_classes = dataset.ntrain_class
    seen_classes = np.array(dataset.seenclasses)

    acc = np.zeros(num_seen_classes)
    for i in range(num_seen_classes):
        acc[i] = (preds[label_T == seen_classes[i]] == seen_classes[i]).mean()
    acc_S_T = acc.mean() * 100

    """  U -> T
    """
    test_labels = np.array(dataset.test_unseen_label)
    #test_images = dataset.test_unseen_feature.cuda()
    test_images = dataset.test_unseen_feature
    preds = np.zeros_like(test_labels)

    start = 0
    while start < test_labels.shape[0]:
        if start + opt.batch_size > test_labels.shape[0]:
            end = test_labels.shape[0]
        else:
            end = start + opt.batch_size

        batch_images = test_images[start:end]

        #batch_ext = batch_images.unsqueeze(0).repeat(class_num, 1, 1)
        #batch_ext = torch.transpose(batch_ext, 0, 1)
        
        batch_ext = unsqu1(batch_images, 0)
        batch_ext = mnp.tile(batch_ext, (class_num, 1, 1))
        input_perm = (1, 0, 2)
        batch_ext = transpose(batch_ext, input_perm)

        # forward
        semantic_proto = APnet(support_attr)
        #semantic_proto_ext = semantic_proto.unsqueeze(0).repeat(end - start, 1, 1)
        semantic_proto_ext = unsqu1(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (end - start, 1, 1))
        
        # relation_pairs = torch.cat([semantic_proto_ext, batch_ext],2).view(-1,dataset.feature_dim + dataset.feature_dim)
        relation_pairs = semantic_proto_ext * batch_ext
        relations = Rnet(relation_pairs).view(-1, class_num)
        #prob = relations.data.cpu().numpy()
        prob = relations.asnumpy()
        pred = np.argmax(prob, 1)
        preds[start:end] = pred

        start = end

    # produce MCA
    label_T = test_labels
    num_unseen_classes = dataset.ntest_class
    unseen_classes = np.array(dataset.unseenclasses)

    acc = np.zeros(num_unseen_classes)
    for i in range(num_unseen_classes):
        acc[i] = (preds[label_T == unseen_classes[i]] == unseen_classes[i]).mean()
    acc_U_T = acc.mean() * 100

    acc = (2 * acc_S_T * acc_U_T) / (acc_S_T + acc_U_T)

    result.acc_list += [acc]
    result.iter_list += [it]
    result.save_model = False
    if acc > result.best_acc:
        result.best_acc = acc
        result.best_iter = it
        result.best_acc_S_T = acc_S_T
        result.best_acc_U_T = acc_U_T
        result.save_model = True

    print("H {:.2f}%  S->T {:.2f}%  U->T {:.2f}%  ".format(acc, acc_S_T, acc_U_T))
    return acc, acc_S_T, acc_U_T


class Result(object):
    def __init__(self):
        self.best_acc = 0.0
        self.best_iter = 0.0
        self.best_acc_S_T = 0.0
        self.best_acc_U_T = 0.0
        self.acc_list = []
        self.iter_list = []


def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        init.xavier_normal(m.weight.data)
        init.constant(m.bias, 0.0)
        
        
def weights_init_ms(net):
    for m in net.cells_and_names():
        if m=='Linear':
            #init.xavier_normal
            XavierUniform(m.weight.data)
            Zero(m.bias)


def reset_grad(nets):
    for net in nets:
        net.zero_grad()


def label2mat(labels, y_dim):
    c = np.zeros([labels.shape[0], y_dim])
    for idx, d in enumerate(labels):
        c[idx, d] = 1
    return c


if __name__ == "__main__":
    train()
