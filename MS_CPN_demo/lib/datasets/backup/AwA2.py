import os, sys, hashlib, torch
from pathlib import Path
import numpy as np
from PIL import Image
from collections import defaultdict
import torch.utils.data as data

class AwA2_Graph(data.Dataset):

  def __init__(self, root, transform, graph_path):
    print("") 

  def __getitem__(self, index):
    print("")

  def __len__(self):
    print("")


class AwA2_Simple(data.Dataset):

  def __init__(self, root, mode, transform=None):
    # This class should load the corresponding AwA2 images or features.
    # Initialize the following variables
    super(AwA2_Simple, self).__init__()
    self.root   = Path(root)
    assert mode in ['train', 'test'], 'invalid mode = {:}'.format(mode)
    if not self.root.exists(): 
      raise RuntimeError('Dataset not found. Please download and generate the dataset first')
    self.transform          = transform
    self.index_feture_label = torch.load( self.root / "idx_fea_label.pth" )['{:}_idx_fea_label'.format(mode)]
    # load every thing belong to this dataset class
    class_index_dict = torch.load( self.root / "cls_idx_dict.pth" )
    self.classNAME2index_wd = class_index_dict['{:}_cls_idx'.format(mode)]
    self.classIDX2name_wd   = class_index_dict['{:}_idx_cls'.format(mode)]
    self.num_classes        = len(self.classIDX2name_wd)
    analysis_count          = defaultdict(lambda: 0)
    self.labels             = []
    for idx in range(len(self.index_feture_label)):
      f_path, (label_index, wordnet_id) = self.index_feture_label[idx]
      analysis_count[label_index] += 1
      self.labels.append( label_index )
    #print (analysis_count)
    #for key, value in self.classIDX2name_wd.items():
    #  print ('Key {:} -> {:}'.format(key, value))
    self.length             = len(self)

  def __getitem__(self, index):
    assert 0 <= index < len(self), 'invalid index = {:}'.format(index)
    feature_path, (label_index, wordnet_id) = self.index_feture_label[index]
    feature = np.load(feature_path)['feat']
    #torch_feature = torch.tensor(feature, dtype=torch.float)
    torch_feature = torch.from_numpy(feature)
    assert label_index == self.labels[index]
    torch_label   = torch.tensor(label_index, dtype=torch.long)
    if self.transform:
      torch_feature = self.transform(torch_feature)
    return torch_feature, torch_label

  def __repr__(self):
    return ('{name}({length:5d} samples with {num_classes} classes)'.format(name=self.__class__.__name__, **self.__dict__))

  def __len__(self):
    return len(self.index_feture_label)

  def loop_class_names(self):
    for i, (key, value) in enumerate(self.classIDX2name_wd.items()):
      assert i == key, 'invalid {:} vs. {:}'.format(i, key)
      yield value


class SubgraphSampler(object):
  def __init__(self, root, mode, classes_per_it, num_samples, iterations): 
    super(SubgraphSampler, self).__init__()
    self.root = Path(root)
    self.label_idx_fea = torch.load( self.root / "idx_fea_label.pth" ) ["{}_label_idx_fea".format(mode)]
    self.classes_per_it  = classes_per_it
    self.sample_per_class = num_samples
    self.iterations = iterations

  def __iter__(self):
    '''
    yield a batch of indexes
    '''
    spc = self.sample_per_class
    cpi = self.classes_per_it

    for it in range(self.iterations):
      print("")  
    

# Test Your Dataset Class
# if __name__ == '__main__':
