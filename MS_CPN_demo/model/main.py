import os, re, sys, time, random, argparse, math, json
import mindspore
import mindspore.nn as nn
from mindspore import set_seed
from mindspore import context
from mindspore import load_checkpoint
import os.path as osp
from mindspore import Tensor
from mindspore.dataset import GeneratorDataset
from mindspore import ops
from mindspore.ops import ExpandDims, Argmax
from mindspore.nn import learning_rate_schedule as lr_schedules

import numpy as np
import scipy.io as sio
from PIL     import ImageFile
from copy    import deepcopy
from pathlib import Path
from semantic_networks import LinearEnsemble,LinearEnsemble1
from relation_networks import PPNRelationNet
from eval_util import evaluate_all
import sklearn.linear_model as models


# This is used to make dirs in lib visiable to your python program
lib_dir = (Path(__file__).parent / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: 
    sys.path.insert(0, str(lib_dir))

from sklearn.cluster import KMeans
from config_utils import Logger, time_string, convert_secs2time, AverageMeter
from config_utils import load_configure, count_parameters_in_MB
from datasets     import ZSHOT_DATA, ZSHOT_DATA_new
from models       import distance_func
from datasets     import MetaSampler
from visuals      import encoder_template

unsqu = ExpandDims()
concat = ops.Concat()


def obtain_relation_models(name, att_dim, image_dim):
  model_name = name.split('-')[0]
  _, att_C, hidden_C, degree, T = name.split('-')
    
  return PPNRelationNet(att_dim, image_dim, int(att_C), int(hidden_C), int(T), int(degree))

def obtain_semantic_models(name, field_centers):
  model_name = name.split('-')[0]
  if model_name == 'Linear':
    _, out_dim = name.split('-')
    return LinearEnsemble(field_centers, int(out_dim))   #out_dim=2048

def obtain_semantic_models1(name, field_centers, num):
  model_name = name.split('-')[0]
  if model_name == 'Linear':
    _, out_dim = name.split('-')
    return LinearEnsemble1(field_centers, int(out_dim), num)



def train_model(xargs, loader, semantics,visual, test_class_att, test_adj_dis, adj_distances, network1,network2,relation_network,network3, optimizer, config, logger, train_classes):
  
  batch_time, Xlosses, accs, end = AverageMeter(), AverageMeter(), AverageMeter(), time.time()
  labelMeter = AverageMeter()
  
  network1.set_train()
  network2.set_train()
  relation_network.set_train()
  network3.set_train()

  logger.print('[TRAIN---{:}], semantics-shape={:}, adj_distances-shape={:}, config={:}'.format(config.epoch_str, semantics.shape, adj_distances.shape, config))

  for data in loader.create_dict_iterator():
    image_feats = data["image_feats"]
    targets = data["targets"]
    targets = targets.asnumpy()
    
    support_labels =  np.array(train_classes)
    re_batch_labels = np.zeros_like(targets)
    train_class_num = support_labels.shape[0]
    

    for i in range(train_class_num):
        re_batch_labels[targets == support_labels[i]] = i
    
    batch_label_set = set(re_batch_labels.tolist()) 
    batch_label_lst = list(batch_label_set)
    class_num       = len(batch_label_lst)
    batch, feat_dim = image_feats.shape
    
    batch_attrs     = semantics[batch_label_lst, :]  #训练类语义
    
    print('bbbbb')
    print(batch_attrs.shape)
    print(batch_label_lst)
    
    batch_vis_center= visual[Tensor(np.array(batch_label_lst))]
    
    print('bbbbb')
    print(batch_vis_center.shape)

    batch_attrs = Tensor(batch_attrs.astype(np.float32)) 
    batch_vis_center = Tensor(batch_vis_center.astype(np.float32))
    print(type(batch_attrs))    
    re_att = network1(batch_attrs)
    re_vis = network2(batch_vis_center)
    relation1 ,att_gcn =  relation_network(image_feats, re_att)
    relation2 ,vis_gcn =  relation_network(image_feats, re_vis)

    new_target_idxs = [batch_label_lst.index(x) for x in re_batch_labels.tolist()]
    new_target_idxs = torch.LongTensor(new_target_idxs)
    one_hot_labels  = torch.zeros(batch, class_num).scatter_(1, new_target_idxs.view(-1,1), 1)
    target__labels  = new_target_idxs.cuda()
    
    feature_loss = nn.MSELoss(reduction='sum')
    cat_loss     = nn.CrossEntropyLoss()
    re_semantic  = network3(att_gcn)
    re_semantic_loss  = feature_loss(re_semantic, batch_attrs.cuda()) / (batch )
    loss4  =  re_semantic_loss
    
    CE = nn.CrossEntropyLoss()
    
    if config.loss_type == 'sigmoid-mse':
      loss1 = CE(relation1 * xargs.cross_entropy_lambda, target__labels)
      loss2 = CE(relation2 * xargs.cross_entropy_lambda, target__labels)
      loss3 = F.mse_loss(torch.sigmoid(re_vis), torch.sigmoid(re_att), reduction='elementwise_mean')
      loss = loss1+xargs.lambda_1*loss2+xargs.lambda_2*loss3+xargs.lambda_3*loss4
    else:
      raise ValueError('invalid loss type : {:}'.format(config.loss_type))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    # analysis
    Xlosses.update(loss.item(), batch)
    predict_labels = torch.argmax(relation1, dim=1)
    with torch.no_grad():
      accuracy = (predict_labels.cpu() == new_target_idxs.cpu()).float().mean().item()
      accs.update(accuracy*100, batch)
      labelMeter.update(class_num, 1)

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()  
  
    if batch_idx % config.log_interval == 0 or batch_idx + 1 == len(loader):
      Tstring = 'TIME[{batch_time.val:.2f} ({batch_time.avg:.2f})]'.format(batch_time=batch_time)
      Sstring = '{:} [{:}] [{:03d}/{:03d}]'.format(time_string(), config.epoch_str, batch_idx, len(loader))
      Astring = 'loss={:.7f} ({:.5f}), acc@1={:.1f} ({:.1f})'.format(Xlosses.val, Xlosses.avg, accs.val, accs.avg)
      logger.print('{:} {:} {:} B={:}, L={:} ({:.1f}) : {:}'.format(Sstring, Tstring, Astring, batch, class_num, labelMeter.avg, batch_label_lst[:3]))
  return Xlosses.avg, accs.avg


def main(xargs):
  # your main function
  # print some necessary informations
  # create logger
  if not os.path.exists(xargs.log_dir):
    os.makedirs(xargs.log_dir)
  logger = Logger(xargs.log_dir, xargs.manual_seed)
  logger.print ("args :\n{:}".format(xargs))

  # set random seed

  random.seed(xargs.manual_seed)
  np.random.seed(xargs.manual_seed)
  set_seed(xargs.manual_seed)
  
  context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

  logger.print('Start Main with this file : {:}, and load {:}.'.format(__file__, xargs.data_root))

  matcontent1   = sio.loadmat('/media/cqu/D/SWL/zero-shot-propagation/datasets/zshots/xlsa17/data/AWA2/res101.mat')
  matcontent2   = sio.loadmat('/media/cqu/D/SWL/zero-shot-propagation/datasets/zshots/xlsa17/data/AWA2/att_splits.mat')
  img_feature    = matcontent1['features'].T
  img_label      = matcontent1['labels'].astype(int).squeeze() - 1
  image_files    = matcontent1['image_files']
  
  allclasses     = [x[0][0] for x in matcontent2['allclasses_names']]
  # get specific information
  trainval_loc        = matcontent2['trainval_loc'].squeeze() - 1
  trainval_feature    = img_feature[trainval_loc]
  trainval_feature    = Tensor(trainval_feature.astype(np.float32))
  trainval_label      = img_label[trainval_loc]
  trainval_classes    = set(trainval_label.tolist())

  test_seen_loc       = matcontent2['test_seen_loc'].squeeze() - 1
  test_seen_feature   = img_feature[test_seen_loc]
  test_seen_feature   = Tensor(test_seen_feature.astype(np.float32))
  test_seen_label     = img_label[test_seen_loc]

  test_unseen_loc     = matcontent2['test_unseen_loc'].squeeze() - 1
  test_unseen_feature = img_feature[test_unseen_loc]
  test_unseen_feature = Tensor(test_unseen_feature.astype(np.float32))
  test_unseen_label   = img_label[test_unseen_loc]
  
  attributes          = Tensor(matcontent2['att'].T.astype(np.float32))
  
  train_classes  = sorted( list(set(trainval_label.tolist())) )
  unseen_classes = sorted( list(set(test_unseen_label.tolist())) )
  
  # All labels return original value between 0-49
  train_dataset       = ZSHOT_DATA(allclasses, trainval_feature, test_seen_feature, test_unseen_feature, trainval_label, test_seen_label, test_unseen_label, 'train')
  
  
  batch_size          = xargs.class_per_it * xargs.num_shot
  total_episode       = ((len(train_dataset) / batch_size) // 100 + 1) * 100
  train_sampler       = MetaSampler(train_dataset, total_episode, xargs.class_per_it, xargs.num_shot)
  
  train_dataset_new   = ZSHOT_DATA_new(allclasses, trainval_feature, test_seen_feature, test_unseen_feature, trainval_label, test_seen_label, test_unseen_label, 'train')
  train_loader_new    = GeneratorDataset(train_dataset_new, ["image_feats", "targets"], num_parallel_workers=1, sampler=train_sampler)
  
  train_loader        = GeneratorDataset(train_dataset, ["batch_idx", "image_feats", "targets"], num_parallel_workers=1, sampler=train_sampler)
  
  test_seen_dataset   = ZSHOT_DATA(allclasses, trainval_feature, test_seen_feature, test_unseen_feature, trainval_label, test_seen_label, test_unseen_label, 'test-seen')
  test_seen_loader    = GeneratorDataset(test_seen_dataset, ["batch_idx", "image_feats", "targets"], sampler=train_sampler)
  
  test_unseen_dataset = ZSHOT_DATA(allclasses, trainval_feature, test_seen_feature, test_unseen_feature, trainval_label, test_seen_label, test_unseen_label, 'test-unseen')
  test_unseen_loader  = GeneratorDataset(test_unseen_dataset, ["batch_idx", "image_feats", "targets"], sampler=train_sampler)
  
  logger.print('train-dataset       : {:}'.format(train_dataset))
  logger.print('test-seen-dataset   : {:}'.format(test_seen_dataset))
  logger.print('test-unseen-dataset : {:}'.format(test_unseen_dataset))


  features       = Tensor(matcontent2['original_att'].T.astype(np.float32))
  train_features = features[train_classes, :]
  test_class_att = features[unseen_classes, :]
  logger.print('feature-shape={:}, train-feature-shape={:}'.format(list(features.shape), list(train_features.shape)))

  kmeans = KMeans(n_clusters=xargs.clusters, random_state=1337).fit(train_features.asnumpy())
  att_centers = Tensor(kmeans.cluster_centers_.astype(np.float32))
  
  for cls in range(xargs.clusters):
    logger.print('[cluster : {:}] has {:} elements.'.format(cls, (kmeans.labels_ == cls).sum()))
  logger.print('Train-Feature-Shape={:}, use {:} clusters, shape={:}'.format(train_features.shape, xargs.clusters, att_centers.shape))

  # build adjacent matrix
  distances     = distance_func(attributes, attributes, 'euclidean-pow')
  xallx_adj_dis = distances
  train_adj_dis = distances[train_classes,:][:,train_classes]
  test_adj_dis  = distances[unseen_classes,:][:,unseen_classes]
  

  target_VC=[]
  train_center=[]
  is_first = True
  for x in train_classes:
    trainval_label= np.array(trainval_label)
    idx = (trainval_label==x).nonzero()[0]
    train_features_all = trainval_feature
    idx = Tensor(idx)
    train_features_1 = train_features_all[idx]
    sum=[0.0]*2048
    sum=Tensor(np.array(sum))
    cnt=0
    for y in train_features_1:
        cnt+=1
        sum=sum+y

    sum/=cnt
    
    avg=Tensor(sum.astype(np.float32))
    unsqu = ExpandDims()
    avg = unsqu(avg, 0)
    
    if is_first:
      is_first=False
      train_center = avg
    else:     
      train_center = concat((train_center, avg))   
    target_VC = train_center
 
  kmeans1 = KMeans(n_clusters=xargs.clusters, random_state=1330).fit(target_VC.asnumpy())
  vis_centers = Tensor(kmeans1.cluster_centers_.astype(np.float32))
  network1= obtain_semantic_models(xargs.semantic_name, att_centers)
  network2= obtain_semantic_models1(xargs.visual_name, vis_centers, xargs.class_per_it)
  relation_network = obtain_relation_models(xargs.relation_name, 2048, 2048)
  print(type(att_centers))
  
  _,vis_dim = vis_centers.shape
  _,att_dim = att_centers.shape


  encoder_vis = encoder_template(vis_dim, xargs.latent_size, xargs.hidden_size_vis, xargs.hidden_size_sem, att_dim)
  network3= encoder_vis
  '''
  parameters = [{'params': list(network1.parameters())},
              {'params': list(network2.parameters())}, 
              {'params': list(relation_network.parameters())},
              {'params': list(network3.parameters())},
              ]
  '''
  parameters = []
  for item in network1.get_parameters():
    parameters.append(item)
  for item in network2.get_parameters():
    parameters.append(item)
  for item in relation_network.get_parameters():
    parameters.append(item)
  for item in network3.get_parameters():
    parameters.append(item)
    
  optimizer  = nn.Adam(parameters, learning_rate=xargs.lr, beta1=0.9, beta2=0.999, eps=1e-08, weight_decay=xargs.weight_decay)
  
  logger.print('optimizer : {:}'.format(optimizer))
  
  model_lst_path  = logger.checkpoint('ckp-last-{:}.pth'.format(xargs.manual_seed))
  if os.path.isfile(str(model_lst_path)):
    checkpoint  = torch.load(model_lst_path)
    start_epoch = checkpoint['epoch'] + 1
    best_accs   = checkpoint['best_accs']
    network.load_state_dict(checkpoint['network'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    logger.print ('load checkpoint from {:}'.format(model_lst_path))
  else:
    start_epoch, best_accs = 0, {'train': -1, 'xtrain': -1, 'zs': -1, 'gzs-seen': -1, 'gzs-unseen': -1, 'gzs-H':-1, 'best-info': None}
  
  epoch_time, start_time = AverageMeter(), time.time()
  # training
  for iepoch in range(start_epoch, xargs.epochs):
    # set some classes as fake zero-shot classes
    time_str = convert_secs2time(epoch_time.val * (xargs.epochs- iepoch), True) 
    epoch_str= '{:03d}/{:03d}'.format(iepoch, xargs.epochs)
  
    config_train = load_configure(None, {'epoch_str': epoch_str, 'log_interval': xargs.log_interval,
                                         'loss_type': xargs.loss_type}, None)

    train_cls_loss, train_acc = train_model(xargs, train_loader_new, train_features,target_VC,test_class_att,test_adj_dis, train_adj_dis, network1,network2,relation_network,network3,optimizer, config_train, logger, train_classes)
    lr_scheduler.step()
    if train_acc > best_accs['train']: best_accs['train'] = train_acc
    logger.print('Train {:} done, cls-loss={:.3f}, accuracy={:.2f}%, (best={:.2f}).\n'.format(epoch_str, train_cls_loss, train_acc, best_accs['train']))

    if iepoch % xargs.test_interval == 0 or iepoch == xargs.epochs -1:
      with torch.no_grad():
        xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
        evaluate_all(epoch_str, train_loader, test_unseen_loader, test_seen_loader, features, xallx_adj_dis, network1,relation_network,  xinfo, best_accs, logger)

    # measure elapsed time
    epoch_time.update(time.time() - start_time)
    start_time = time.time()
  
  # the final evaluation
  logger.print('final evaluation --->>>')
  with torch.no_grad():
    xinfo = {'train_classes' : graph_info['train_classes'], 'unseen_classes': graph_info['unseen_classes']}
    evaluate_all('final-eval', train_loader, test_unseen_loader, test_seen_loader, features, xallx_adj_dis, network1,relation_network, xinfo, best_accs, logger)
  logger.print('-'*200)
  logger.close()

           
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--log_dir' ,       type=str, default='./log_dir',help='Save dir.')
  parser.add_argument('--data_root' ,     type=str,  default='/data_dir' ,help='dataset root')
  parser.add_argument('--loss_type',      type=str,  default='sigmoid-mse',             help='The loss type.')
  parser.add_argument('--semantic_name',  type=str,   default='Linear-2048')
  parser.add_argument('--visual_name',    type=str,   default='Linear-2048')
  parser.add_argument('--relation_name',  type=str,   default='PPN-256-2048-40-10')
  parser.add_argument('--clusters',       type=int,   default=3)
  parser.add_argument('--class_per_it',   type=int,   default=30)
  parser.add_argument('--num_shot'    ,   type=int,   default=1)
  parser.add_argument('--epochs',         type=int,   default=500)
  parser.add_argument('--manual_seed',    type=int,   default=34194)
  parser.add_argument('--lr',             type=float,  default=0.00002)
  parser.add_argument('--weight_decay',   type=float,  default=0.0001)
  parser.add_argument('--num_workers',    type=int,   default= 8,     help='The number of workers.')
  parser.add_argument('--log_interval',   type=int,   default=100,     help='The log-print interval.')
  parser.add_argument('--test_interval',  type=int,   default=5,     help='The evaluation interval.')
  parser.add_argument('--latent_size' ,   type=int,   default=256)
  parser.add_argument('--latent_size1' ,   type=int,   default=85)
  parser.add_argument('--latent_size2' ,   type=int,   default=2048)
  parser.add_argument('--drop_out',       type=float, default=0.5)
  parser.add_argument('--hidden_size_vis', default=[1024])
  parser.add_argument('--hidden_size_sem', default=[256])
  parser.add_argument('--cross_entropy_lambda',    type=float, default=0.01)
  parser.add_argument('--lambda_1',       type=float, default=0.0001)
  parser.add_argument('--lambda_2',       type=float, default=0.001)
  parser.add_argument('--lambda_3',       type=float, default=0.0001)
  args = parser.parse_args()

  if args.manual_seed is None or args.manual_seed < 0:
    args.manual_seed = random.randint(1, 100000)
  assert args.log_dir is not None, 'The log_dir argument can not be None.'
  main(args)