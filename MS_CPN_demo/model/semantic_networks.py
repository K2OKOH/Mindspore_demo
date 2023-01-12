#import torch.nn as nn
#import torch.nn.functional as F
#import torch
import mindspore
from mindspore import nn
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeUniform
from mindspore import ops
import math
import numpy as np

size = ops.Size()

class LinearModule(nn.Cell):

  def __init__(self, field_center, out_dim):
    super(LinearModule, self).__init__()
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.fc = nn.Dense(size(field_center), out_dim)
    self.relu = nn.ReLU()
    self.field_center = field_center


  def construct(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    response = self.relu(self.fc(input_offsets))
    return response


class LinearEnsemble(nn.Cell):

  def __init__(self, field_centers, out_dim):
    super(LinearEnsemble, self).__init__()
    self.individuals = nn.CellList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    self.require_adj = False
    for i in range(field_centers.shape[0]):
      layer = LinearModule(field_centers[i], out_dim)
      self.individuals.append( layer )
    print('aaaaa')
    print(self.individuals)
    self.cluster = field_centers.shape[0]

  def construct(self, semantic_vec):
    responses = []
    for i in range(self.cluster):
      indiv = self.individuals[i]
      responses = responses.append(indiv(semantic_vec))
    feature_anchor = responses[0]
    for i in range(self.cluster):
      if(i==0):
        feature_anchor = responses[0]
      else:
        feature_anchor += responses[i]
    return feature_anchor
    
    
    
class LinearModule1(nn.Cell):

  def __init__(self, field_center, out_dim, num):
    super(LinearModule1, self).__init__()
    
    assert field_center.dim() == 1, 'invalid shape : {:}'.format(field_center.shape)
    self.fc = nn.Dense(size(field_center),out_dim)
    self.w      = Parameter(Tensor(np.ones((size(field_center), out_dim)), mindspore.float32))
    self.b      = Parameter(Tensor(np.ones((num, out_dim)), mindspore.float32))
    self.w = initializer(HeUniform(), [size(field_center), out_dim], mindspore.float32)
    self.b = initializer(HeUniform(), [num, out_dim], mindspore.float32)
    self.relu = nn.ReLU()
    self.field_center = field_center

  def construct(self, semantic_vec):
    input_offsets = semantic_vec - self.field_center
    input_offsets=self.relu(input_offsets+self.b)
    
    return input_offsets


class LinearEnsemble1(nn.Cell):

  def __init__(self, field_centers, out_dim, num):
    super(LinearEnsemble1, self).__init__()
    self.individuals = nn.CellList()
    assert field_centers.dim() == 2, 'invalid shape : {:}'.format(field_centers.shape)
    self.out_dim     = out_dim
    self.require_adj = False
    for i in range(field_centers.shape[0]):
      layer = LinearModule1(field_centers[i], out_dim, num)
      self.individuals.append( layer )
    self.cluster = field_centers.shape[0]
    
  def construct(self, semantic_vec):
    responses = []
    for i in range(self.cluster):
      indiv = self.individuals[i]
      responses = responses.append(indiv(semantic_vec))
    feature_anchor = responses[0]
    for i in range(self.cluster):
      if(i==0):
        feature_anchor = responses[0]
      else:
        feature_anchor += responses[i]
    return feature_anchor