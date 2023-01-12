import time
import numpy as np
import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from scipy.spatial.distance import pdist

pow = ops.Pow()
def distance_func(x, y, xtype):
  if xtype == 'euclidean':
    return euclidean_dist(x, y, True)
  elif xtype == 'euclidean-pow':
    return euclidean_dist(x, y, False)
  elif xtype == 'cosine':
    return cosine_dist(x, y)
  else:
    return dotprod_dist(x, y)


def euclidean_dist(x, y, sqrt=False):
  batchx, dimx = x.shape
  batchy, dimy = y.shape
  x = x.view(batchx, 1, dimx)
  y = y.view(1, batchy, dimy)
  distances = pow(x - y, 2).sum(-1)
  if sqrt:
    distances = distances.clamp(min=1e-12).sqrt()  # for numerical stability
  return distances


def cosine_dist(x, y):
  batchx, dimx = x.shape
  batchy, dimy = y.shape
  distances = pdist.cosine(x,y)
  return distances


def dotprod_dist(x, y):
  batchx, dimx = x.shape
  batchy, dimy = y.shape
  x = x.view(batchx, 1, dimx)
  y = y.view(1, batchy, dimy)
  distances = x * y.sum(axis=2)
  return distances
