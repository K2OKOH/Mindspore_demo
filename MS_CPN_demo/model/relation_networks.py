#import torch.nn as nn
#import torch.nn.functional as F
import mindspore
from mindspore import nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeUniform
import math
#import torch
from distance_utils import distance_func
from mindspore import ops
import mindspore.numpy as np
from mindspore.ops import ExpandDims
matmul = ops.MatMul()
oneslike = ops.OnesLike()
softmax = ops.Softmax(axis=1)
import mindspore.numpy as mnp


class PPNRelationNet(nn.Cell):
  """docstring for RelationNetwork"""
  def __init__(self, att_dim, image_dim, att_hC, hidden_C, T, degree):
    super(PPNRelationNet, self).__init__()
    print(att_dim,image_dim,att_hC,hidden_C)
    
    self.att_g     = initializer(HeUniform(), [att_dim, att_hC], mindspore.float32)
    
    self.T         = T
    assert degree >= 0 and degree < 100, 'invalid degree : {:}'.format(degree)
    self.degree    = degree
    self.thresh    = math.cos(math.pi*degree/180)
    self.img_w     = initializer(HeUniform(), [image_dim, hidden_C], mindspore.float32)
    self.sem_w     = initializer(HeUniform(), [image_dim, hidden_C], mindspore.float32)
    self.sem_b     = initializer(HeUniform(), [1, hidden_C], mindspore.float32)
    
    self.fc        = nn.Dense(hidden_C, 1)
     
    self.fc1 = nn.Dense(att_dim, hidden_C)
    self.relu = nn.ReLU()
    
  def extra_repr(self):
    xshape = 'att-shape: {:}'.format(list(self.att_g.shape)) + ' img-w-shape : {:}'.format(list(self.img_w.shape)) + ' sem-w-shape : {:}'.format(list(self.sem_w.shape))
    return ('{name}(degree={degree:}, thresh={thresh:.3f}, temperature={T:}, '.format(name=self.__class__.__name__, **self.__dict__) + xshape + ')')

  def get_new_attribute(self, attributes):
    att_prop_g = matmul(attributes, self.att_g)
    att_prop_h = matmul(attributes, self.att_g)
    distances  = distance_func(att_prop_g, att_prop_h, 'cosine')
    zero_vec   = -9e15 * oneslike(distances)
    raw_attss  = np.where(distances > self.thresh, distances, zero_vec)
    attention  = softmax(raw_attss * self.T)
    att_outs   = matmul(attention, attributes)
    return att_outs, distances > self.thresh
  

  def construct(self, image_feats, attributes):
    # attribute propgation
    cls_num, at_dim = attributes.shape
    att_outs, _ = self.get_new_attribute(attributes)
    
    batch, feat_dim = image_feats.shape
    image_feats_ext = image_feats.view(batch, 1, -1)
    att_outs        = att_outs.view(1, cls_num, -1)
    print(image_feats_ext.shape)
    unsqu = ExpandDims()
    image_feats_ext = unsqu(image_feats_ext, 1)
    image_feats_ext = mnp.tile(image_feats_ext, (1,cls_num, 1))
    
    att_feats_ext   = unsqu(att_outs, 0)
    att_feats_ext   = mnp.tile(att_feats_ext, (batch,1, 1))
    hidden_feats    = self.relu(ops.matmul(image_feats_ext, self.img_w) + ops.matmul(att_feats_ext, self.sem_w) + self.sem_b )
    outputs         = self.fc(hidden_feats)
    return outputs.view(batch, cls_num),att_outs