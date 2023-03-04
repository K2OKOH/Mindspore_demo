# noinspection PyUnresolvedReferences
import mindspore
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean,
                                       _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer

from mindspore.ops import ExpandDims, Argmax
import mindspore.numpy as mnp
from mindspore import ops


unsqu = ExpandDims()
sigmoid = nn.Sigmoid()
concat = ops.Concat()
class APnetWithLossCell(nn.Cell):
    def __init__(self, APnet, loss_fn, auto_prefix=True):
        super(APnetWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.APnet = APnet
        self.loss_fn = loss_fn

    def construct(self, att_real, vis_real):
        vis = self.APnet(att_real)
        #fake_out = self.netD(fake_data)
        loss_G = self.loss_fn(sigmoid(vis), sigmoid(vis_real))

        return loss_G

class RnetWithLossCell(nn.Cell):
    def __init__(self, APnet, Rnet, loss_fn, class_num, train_class_num, auto_prefix=True):
        # def __init__(self, APnet, Rnet, loss_fn, class_num, train_class_num, lambda_3, auto_prefix=True):
        super(RnetWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.APnet = APnet
        self.Rnet = Rnet
        self.loss_fn = loss_fn
        self.class_num = class_num
        self.train_class_num = train_class_num
        # self.lambda_3 = lambda_3

    def construct(self, support_attr, batch_ext, one_hot_labels, sim_labels):
        semantic_proto = self.APnet(support_attr)
        semantic_proto_ext = unsqu(semantic_proto, 0)
        semantic_proto_ext = mnp.tile(semantic_proto_ext, (one_hot_labels.shape[0], 1, 1))
        
        relation_pairs = semantic_proto_ext * batch_ext
        relations = self.Rnet(relation_pairs).view(-1, self.class_num)
        relation_train = relations[:, :self.train_class_num]
        relation_test = relations[:, self.train_class_num:]
        
        loss1 = self.loss_fn(relation_train, one_hot_labels)
        loss2 = self.loss_fn(relation_test, sim_labels)
        # loss = loss1+self.lambda_3* loss2
        loss = loss1 + 0.01* loss2
        return loss
        
class ReconnetWithLossCell(nn.Cell):
    def __init__(self, APnet, Reconnet, loss_fn, auto_prefix=True):
        super(ReconnetWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.APnet = APnet
        self.Reconnet = Reconnet
        self.loss_fn = loss_fn

    def construct(self, att_real):
        vis = self.APnet(att_real)
        att_fake = self.Reconnet(vis)
        loss_D = self.loss_fn(sigmoid(att_fake), sigmoid(att_real))
        return loss_D

class AllNetWithLossCell(nn.Cell):
    def __init__(self, all_net, loss_bce, loss_mse, class_num, train_class_num, auto_prefix=True):
        super(AllNetWithLossCell, self).__init__(auto_prefix=auto_prefix)
        self.all_net = all_net
        self.loss_bce = loss_bce
        self.loss_mse = loss_mse
        self.class_num = class_num
        self.train_class_num = train_class_num

    def construct(self, batch_images, batch_atts, visual_batch_val, att_batch_val, support_attr, batch_ext, one_hot_labels, sim_labels):

        semantic_proto_batch, rec_sem, rec_sem_unseen, relations, unseen_semantic_proto_batch \
            = self.all_net(batch_atts, att_batch_val, one_hot_labels, support_attr, batch_ext)
        
        relations = relations.view(-1, self.class_num)
        relation_train = relations[:, :self.train_class_num]
        relation_test = relations[:, self.train_class_num:]
        
        loss1 = self.loss_bce(relation_train, one_hot_labels)
        loss2 = self.loss_bce(relation_test, sim_labels)
        loss3 = self.loss_mse(sigmoid(semantic_proto_batch), sigmoid(batch_images))
        loss4 = self.loss_mse(sigmoid(rec_sem), sigmoid(batch_atts))
        loss5 = self.loss_mse(sigmoid(unseen_semantic_proto_batch), sigmoid(visual_batch_val))
        loss6 = self.loss_mse(sigmoid(rec_sem_unseen), sigmoid(att_batch_val))
        loss = loss1 + 0.001 * loss2+ 0.1 * loss3+ 0.1 * loss5 + 0.1 * loss4 + 0.1 * loss6
        
        return loss

'''
class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        APnet_with_loss,
        Rnet_with_loss,
        Reconnet_with_loss,
        optimizerAP: nn.Optimizer,
        optimizerR: nn.Optimizer,
        optimizerRR: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.APnet = APnet_with_loss
        self.APnet.set_grad()
        self.APnet.add_flags(defer_inline=True)

        self.Rnet = Rnet_with_loss
        self.Rnet.set_grad()
        self.Rnet.add_flags(defer_inline=True)
        
        self.Reconnet = Reconnet_with_loss
        self.Reconnet.set_grad()
        self.Reconnet.add_flags(defer_inline=True)

        self.weights_APnet = optimizerAP.parameters
        self.optimizerAPnet = optimizerAP
        self.weights_Rnet = optimizerR.parameters
        self.optimizerR = optimizerR
        self.weights_Reconnet = optimizerRR.parameters
        self.optimizerRR = optimizerRR

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer_APnet = F.identity
        self.grad_reducer_Rnet  = F.identity
        self.grad_reducer_Reconnet  = F.identity
        #self.parallel_mode = _get_parallel_mode()
        #if self.parallel_mode in (ParallelMode.DATA_PARALLEL,
                                 # ParallelMode.HYBRID_PARALLEL):
            #self.reducer_flag = True
        #if self.reducer_flag:
            #mean = _get_gradients_mean()
            #degree = _get_device_num()
            #self.grad_reducer_G = DistributedGradReducer(self.weights_G, mean, degree)
            #self.grad_reducer_D = DistributedGradReducer(self.weights_D, mean, degree)

    def trainAPnet(self, att, vis, loss, loss_net, grad, optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        #grads = grad(loss_net, weights)(real_data, latent_code, sens)
        #grads = grad(loss_net, weights)
        grads = grad(loss_net, weights)(att, vis, sens)
        grads = grad_reducer(grads)
        return F.depend(loss, optimizer(grads))

    def trainRnet(self, support_attr, batch_ext, one_hot_labels, sim_labels, loss, loss_net, grad, optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        #grads = grad(loss_net, weights)(latent_code, sens)
        grads = grad(loss_net, weights)(support_attr, batch_ext, one_hot_labels, sim_labels, sens)
        grads = grad_reducer(grads)
        return F.depend(loss, optimizer(grads))
        
    def trainReconnet(self, att, loss, loss_net, grad, optimizer, weights, grad_reducer):
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        #grads = grad(loss_net, weights)(latent_code, sens)
        grads = grad(loss_net, weights)(att, sens)
        grads = grad_reducer(grads)
        return F.depend(loss, optimizer(grads))

    # def construct(self, lambda_1, lambda_2, lambda_3, batch_images, batch_atts, visual_batch_val, att_batch_val, one_hot_labels, sim_labels, support_attr, batch_ext):
    def construct(self, batch_images, batch_atts, visual_batch_val, att_batch_val, one_hot_labels, sim_labels, support_attr, batch_ext):
        #loss3 = self.APnet(batch_atts, batch_images)
        #loss5 = self.APnet(att_batch_val, visual_batch_val)
        
        att = concat((batch_atts, att_batch_val))
        vis = concat((batch_images, visual_batch_val))
        
        loss35 = self.APnet(att, vis)
        
        #loss1 = self.Rnet(support_attr, batch_ext, one_hot_labels)
        #loss2 = self.Rnet(support_attr, batch_ext, sim_labels)
        loss1 = self.Rnet(support_attr, batch_ext, one_hot_labels, sim_labels)
        
        #loss4 = self.Reconnet(batch_atts)
        #loss7 = self.Reconnet(att_batch_val)
        
        loss47 = self.Reconnet(att)
        
        #loss = loss1 + lambda_1*loss3+  lambda_1*loss5 + lambda_2*loss4 + lambda_2*loss7
        # loss = loss1 + lambda_1*loss35+  lambda_2*loss47 
        loss = loss1 + 0.1*loss35+  0.1*loss47 

        out1 = self.trainAPnet(att, vis, loss, self.APnet,
                            self.grad, self.optimizerAPnet, self.weights_APnet,
                            self.grad_reducer_APnet)
        out2 = self.trainRnet(support_attr, batch_ext, one_hot_labels, sim_labels, loss, self.Rnet, self.grad,
                            self.optimizerR, self.weights_Rnet,
                            self.grad_reducer_Rnet)
        out3 = self.trainReconnet(att, loss, self.Reconnet, self.grad,
                            self.optimizerRR, self.weights_Reconnet,
                            self.grad_reducer_Reconnet)

        # 反向传播的缩放系数
        # sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        # grads = self.grad(self.net_loss, self.weights)(x1, x2, y1, y2, sens)
        # grads = self.grad_reducer(grads)
        # loss = F.depend(loss, self.optimizer(grads))

        return loss, out1, out2, out3
'''

class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        all_net_with_loss,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.net = all_net_with_loss
        self.net.set_grad()
        # self.net.add_flags(defer_inline=True)

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, batch_images, batch_atts, visual_batch_val, att_batch_val, one_hot_labels, sim_labels, support_attr, batch_ext):

        loss = self.net(batch_images, batch_atts, visual_batch_val, att_batch_val, support_attr, batch_ext, one_hot_labels, sim_labels)

        # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(batch_images, batch_atts, visual_batch_val, att_batch_val, support_attr, batch_ext, one_hot_labels, sim_labels, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))

        # return loss, out1, out2, out3
        return loss