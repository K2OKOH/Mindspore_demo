import numpy as np
import scipy.io as sio
# noinspection PyUnresolvedReferences
#from termcolor import cprint
from sklearn import preprocessing
#import torch
from sklearn.metrics.pairwise import cosine_similarity
# noinspection PyUnresolvedReferences
#import torch.nn.functional as F
from mindspore import Tensor


def map_label(label, classes):
    mapped_label = np.zeros_like(label)
    for i in range(classes.shape[0]):
        if sum(label == classes[i])==0:
            continue
        else:
            mapped_label[label == classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            if opt.dataset == 'imageNet1K':
                self.read_matimagenet(opt)
            else:
                self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.feature_dim = self.train_feature.shape[1]
        self.att_dim = self.attribute.shape[1]
        self.text_dim = self.att_dim
        self.train_cls_num = self.seenclasses.shape[0]
        self.test_cls_num = self.unseenclasses.shape[0]
        #self.tr_cls_centroid = np.zeros([self.seenclasses.shape[0], self.feature_dim], np.float32)  # .astype(np.float32)
        #for i in range(self.seenclasses.shape[0]):
            #self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].numpy(), axis=0)
            #self.tr_cls_centroid[i] = np.mean(self.train_feature[self.train_label == i].asnumpy(), axis=0)

    def read_matimagenet(self, opt):
        if opt.preprocessing:
            print('MinMaxScaler...')
            scaler = preprocessing.MinMaxScaler()
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = scaler.fit_transform(np.array(matcontent['features']))
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = scaler.transform(np.array(matcontent['features_val']))
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()
            matcontent = h5py.File('/BS/xian/work/data/imageNet21K/extract_res/res101_1crop_2hops_t.mat', 'r')
            feature_unseen = scaler.transform(np.array(matcontent['features']))
            label_unseen = np.array(matcontent['labels']).astype(int).squeeze() - 1
            matcontent.close()
        else:
            matcontent = h5py.File(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat", 'r')
            feature = np.array(matcontent['features'])
            label = np.array(matcontent['labels']).astype(int).squeeze() - 1
            feature_val = np.array(matcontent['features_val'])
            label_val = np.array(matcontent['labels_val']).astype(int).squeeze() - 1
            matcontent.close()

        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + ".mat")
        #self.attribute = torch.from_numpy(matcontent['w2v']).float()
        #self.train_feature = torch.from_numpy(feature).float()
        #self.train_label = torch.from_numpy(label).long()
        #self.test_seen_feature = torch.from_numpy(feature_val).float()
        #self.test_seen_label = torch.from_numpy(label_val).long()
        #self.test_unseen_feature = torch.from_numpy(feature_unseen).float()
        #self.test_unseen_label = torch.from_numpy(label_unseen).long()
        self.attribute = Tensor(matcontent['w2v']).astype(np.float32)
        self.train_feature = Tensor(feature).astype(np.float32)
        self.train_label = Tensor(label).astype(np.long)
        self.test_seen_feature = Tensor(feature_val).astype(np.float32)
        self.test_seen_label = Tensor(label_val).astype(np.long)
        self.test_unseen_feature = Tensor(feature_unseen).astype(np.float32)
        self.test_unseen_label = Tensor(label_unseen).astype(np.long)
        self.ntrain = self.train_feature.size()[0]
        #self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        #self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        #self.train_class = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.seenclasses = Tensor(np.unique(self.train_label.asnumpy().astype(np.int32)))
        self.unseenclasses = Tensor(np.unique(self.test_unseen_label.asnumpy().astype(np.int32)))
        self.train_class = Tensor(np.unique(self.train_label.asnumpy())).astype(np.int32)
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)

    def read_matdataset(self, opt):
        #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        matcontent = sio.loadmat('./data/CUB1_data/res101.mat')
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        #matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        matcontent = sio.loadmat('./data/CUB1_data/original_att_splits.mat')
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        #self.attribute = torch.from_numpy(matcontent['att'].T).float()
        #self.attribute = Tensor(matcontent['att'].T).astype(np.float32)

        self.attribute = Tensor(matcontent['att'].T.astype(np.float32))
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                #self.train_feature = torch.from_numpy(_train_feature).float()
                self.train_feature = Tensor(_train_feature).astype(np.float32)
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                #self.train_label = torch.from_numpy(label[trainval_loc]).long()
                #self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.train_label = Tensor(label[trainval_loc].astype(np.int32))
                self.test_unseen_feature = Tensor(_test_unseen_feature).astype(np.float32)
                self.test_unseen_feature.mul_(1 / mx)
                #self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                #self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_unseen_label = Tensor(label[test_unseen_loc]).astype(np.long)
                self.test_seen_feature = Tensor(_test_seen_feature).astype(np.float32)
                self.test_seen_feature.mul_(1 / mx)
                #self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                self.test_seen_label = Tensor(label[test_seen_loc]).astype(np.long)
            else:
                #self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                #self.train_label = torch.from_numpy(label[trainval_loc]).long()
                #self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                #self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                #self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                #self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                self.train_feature = Tensor(feature[trainval_loc].astype(np.float32))
                self.train_label = Tensor(label[trainval_loc].astype(np.int32))
                self.test_unseen_feature = Tensor(feature[test_unseen_loc].astype(np.float32))
                self.test_unseen_label = Tensor(label[test_unseen_loc]).astype(np.long)
                self.test_seen_feature = Tensor(feature[test_seen_loc].astype(np.float32))
                self.test_seen_label = Tensor(label[test_seen_loc]).astype(np.long)
        else:
            #self.train_feature = torch.from_numpy(feature[train_loc]).float()
            #self.train_label = torch.from_numpy(label[train_loc]).long()
            #self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            #self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()
            self.train_feature = Tensor(feature[train_loc]).astype(np.float32)
            self.train_label = Tensor(label[train_loc].astype(np.int32))
            self.test_unseen_feature = Tensor(feature[val_unseen_loc].astype(np.float32))
            self.test_unseen_label = Tensor(label[val_unseen_loc].astype(np.int32))

        #self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        #self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.seenclasses = Tensor(np.unique(self.train_label.asnumpy().astype(np.int32)))
        self.unseenclasses = Tensor(np.unique(self.test_unseen_label.asnumpy().astype(np.int32)))
        #print(self.seenclasses)
        #print(self.unseenclasses)
        #self.ntrain = self.train_feature.size()[0]
        #self.ntrain_class = self.seenclasses.size(0)
        #self.ntest_class = self.unseenclasses.size(0)
        #self.train_class = self.seenclasses.clone()
        self.ntrain = self.train_feature.shape[0]
        self.ntrain_class = self.seenclasses.shape[0]
        self.ntest_class = self.unseenclasses.shape[0]

        self.train_class = self.seenclasses
        #self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()
        self.allclasses = Tensor(np.arange(0, self.ntrain_class + self.ntest_class)).astype(np.long)

        
        self.train_att = self.attribute[self.seenclasses].asnumpy()
        self.test_att  = self.attribute[self.unseenclasses].asnumpy()
        
        sim_re_2_A  =  np.linalg.inv(np.dot(self.test_att, self.test_att.transpose())+opt.beta*np.identity(self.test_att.shape[0]))
        sim_re_2_B  =  np.dot(sim_re_2_A,  self.test_att)
        sim_re_2    =  np.dot(sim_re_2_B,  self.train_att.transpose())
        sim_re_2[sim_re_2<0.001] = 0
        #tmp2_2  = np.sum(sim_re_2, axis=1)
        #tmp3_2= np.tile(tmp2_2, (sim_re_2.shape[1],1)).transpose()
        #sim_re_2 = sim_re_2/ tmp3_2
        #self.sim_re_2     =  torch.from_numpy(sim_re_2.transpose()).float()
        self.sim_re_2     =  Tensor(sim_re_2.transpose()).astype(np.float32)




class FeatDataLayer(object):   # by Ethan provide the ROI feature data for ZSL learning.
    def __init__(self, label, feat_data,  opt):
        """Set the roidb to be used by this layer during training."""
        #self._roidb = roidb
        assert len(label) == feat_data.shape[0]
        self._opt = opt
        self._feat_data = feat_data
        self._label = label
        self._shuffle_roidb_inds()
        self._epoch = 0
    def _shuffle_roidb_inds(self):
        """Randomly permute the training roidb."""
        self._perm = np.random.permutation(np.arange(len(self._label)))
        # self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""

        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batchsize

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        new_epoch = False
        if self._cur + self._opt.batch_size >= len(self._label):
            self._shuffle_roidb_inds()
            self._epoch += 1
            new_epoch = True

        db_inds = self._perm[self._cur:self._cur + self._opt.batch_size]
        self._cur += self._opt.batch_size

        minibatch_feat = np.array([self._feat_data[i] for i in db_inds])
        minibatch_label = np.array([self._label[i] for i in db_inds])
        blobs = {'data': minibatch_feat, 'labels':minibatch_label, 'newEpoch':new_epoch}
        return blobs

    def forward(self):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()
        return blobs

    def get_whole_data(self):
        blobs = {'data': self._feat_data, 'labels': self._label}
        return blobs
