# noinspection PyUnresolvedReferences
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, XavierUniform, Zero
import mindspore.numpy as mnp

rdc_text_dim = 1000
z_dim = 100
h_dim = 1024

class _param:
    def __init__(self):
        self.rdc_text_dim = rdc_text_dim
        self.z_dim = z_dim
        self.h_dim = h_dim

class _AttributeNet(nn.Cell):
    """docstring for RelationNetwork"""
    def __init__(self,input_size,output_size):
        super(_AttributeNet, self).__init__()
        self.fc1 = nn.Dense(input_size,h_dim)
        self.fc2 = nn.Dense(h_dim,output_size)
        self.relu = nn.LeakyReLU()

    def construct(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
class _RelationNet(nn.Cell):
    """docstring for RelationNetwork"""
    def __init__(self,input_size):
        super(_RelationNet, self).__init__()
        self.fc1 = nn.Dense(input_size,h_dim)
        self.fc2 = nn.Dense(h_dim,1)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.Sigmoid()

    def construct(self,x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return x

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

# 现画网络结构图
class All_Net(nn.Cell):
    """docstring for RelationNetwork"""
    def __init__(self, APnet, Rnet, Reconnet):
        super(All_Net, self).__init__()
        self.APnet = APnet
        self.Rnet = Rnet
        self.Reconnet = Reconnet

        self.concat = ops.Concat()
        self.unsqu = ops.ExpandDims()

    def construct(self, batch_attr, att_batch_val, one_hot_labels, support_attr, batch_ext):
        
        semantic_proto = self.APnet(support_attr)
        semantic_proto_batch = self.APnet(batch_attr)
        semantic_proto_ext = self.unsqu(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (one_hot_labels.shape[0], 1, 1))
        
        relation_pairs = semantic_proto_ext * batch_ext
        relations = self.Rnet(relation_pairs)
        
        unseen_semantic_proto_batch = self.APnet(att_batch_val)

        rec_sem = self.Reconnet(semantic_proto_batch)
        rec_sem_unseen = self.Reconnet(unseen_semantic_proto_batch)
        

        return semantic_proto_batch, rec_sem, rec_sem_unseen, relations, unseen_semantic_proto_batch