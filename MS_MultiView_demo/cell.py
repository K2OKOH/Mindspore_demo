# noinspection PyUnresolvedReferences
import mindspore
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C

from mindspore.ops import ExpandDims, Argmax
import mindspore.numpy as mnp
from mindspore import ops

from mindspore import log as logger

class_num = 10

class CE_net(nn.Cell):
    def __init__(self, G1_C_net, loss_fn, auto_prefix=False):
        super(CE_net, self).__init__(auto_prefix=False)
        self.net = G1_C_net
        self.loss_fn = loss_fn

    def construct(self, x, y):
        feat = self.net(x)
        # print(feat)
        loss_ce = self.loss_fn(feat, y)

        return loss_ce

class DANN_loss_net(nn.Cell):
    def __init__(self, net, loss_fn, auto_prefix=False):
        super(DANN_loss_net, self).__init__(auto_prefix=False)
        self.net = net
        self.loss_fn = loss_fn
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()

    def construct(self, x1, x2, y1, y2):
        cls_s1, dom_s1 = self.net(x1)
        cls_s2, dom_s2 = self.net(x2)
        # print(feat)
        loss_cls= self.loss_fn(cls_s1, y1) + self.loss_fn(cls_s2, y2)
        domain_s1 = self.zeros(dom_s1.shape[0], mindspore.int64)
        domain_s2 = self.ones(dom_s2.shape[0], mindspore.int64)
        loss_dc = self.loss_fn(dom_s1, domain_s1) + self.loss_fn(dom_s2, domain_s2)

        loss = loss_cls + loss_dc

        return loss

class MV3_loss_net(nn.Cell):
    def __init__(self, net, ce_loss, mse_loss, auto_prefix=False):
        super(MV3_loss_net, self).__init__(auto_prefix=False)
        self.net = net
        self.ce_loss = ce_loss
        self.mse_loss = mse_loss
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()

    def construct(self, x1, x2, y1, y2):
        class_out_s1, feat_s1, rc1_s1, rc2_s1, rc3_s1, \
        cp1_s1, cp2_s1, cp3_s1, \
        dom1_s1, dom2_s1, dom3_s1 \
            = self.net(x1)

        class_out_s2, feat_s2, rc1_s2, rc2_s2, rc3_s2, \
        cp1_s2, cp2_s2, cp3_s2, \
        dom1_s2, dom2_s2, dom3_s2 \
            = self.net(x2)
        
        # 分类损失
        loss_cls = self.ce_loss(class_out_s1, y1) + self.ce_loss(class_out_s2, y2)
        # 领域判别损失
        domain_s1 = self.zeros(dom1_s1.shape[0], mindspore.int64)
        domain_s2 = self.ones(dom1_s2.shape[0], mindspore.int64)
        loss_dc1_s1 = self.ce_loss(dom1_s1, domain_s1)
        loss_dc1_s2 = self.ce_loss(dom1_s2, domain_s2)
        loss_dc2_s1 = self.ce_loss(dom2_s1, domain_s1)
        loss_dc2_s2 = self.ce_loss(dom2_s2, domain_s2)
        loss_dc3_s1 = self.ce_loss(dom3_s1, domain_s1)
        loss_dc3_s2 = self.ce_loss(dom3_s2, domain_s2)
        loss_dc = loss_dc1_s1 + loss_dc1_s2 + loss_dc2_s1 + loss_dc2_s2 + loss_dc3_s1 + loss_dc3_s2 
        # 重构损失
        loss_rc1_s1 = self.mse_loss(rc1_s1, feat_s1)
        loss_rc1_s2 = self.mse_loss(rc1_s2, feat_s2)
        loss_rc2_s1 = self.mse_loss(rc2_s1, feat_s1)
        loss_rc2_s2 = self.mse_loss(rc2_s2, feat_s2)
        loss_rc3_s1 = self.mse_loss(rc3_s1, feat_s1)
        loss_rc3_s2 = self.mse_loss(rc3_s2, feat_s2)
        loss_rc = loss_rc1_s1 + loss_rc1_s2 + loss_rc2_s1 + loss_rc2_s2 + loss_rc3_s1 + loss_rc3_s2
        # 视角损失
        dif12_s1 = self.mse_loss(cp1_s1, cp2_s1)
        dif12_s2 = self.mse_loss(cp1_s2, cp2_s2)
        dif23_s1 = self.mse_loss(cp3_s1, cp2_s1)
        dif23_s2 = self.mse_loss(cp3_s2, cp2_s2)
        dif13_s1 = self.mse_loss(cp1_s1, cp3_s1)
        dif13_s2 = self.mse_loss(cp1_s2, cp3_s2)
        loss_dif = dif12_s1 + dif12_s2 + dif23_s1 + dif23_s2 + dif13_s1 + dif13_s2
        # print(feat)
        loss = loss_cls + 0.01 * (loss_dc + loss_rc + 0.1 * loss_dif)

        return loss

# 单个cell的调试
class TrainOneStepCell(nn.Cell):
    def __init__(
        self,
        CE_net,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.CEnet = CE_net
        self.CEnet.set_grad()
        # self.CEnet.add_flags(defer_inline=True)

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x, y):
        loss = self.CEnet(x, y)
        # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.CEnet, self.weights)(x, y, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_PACS(nn.Cell):
    def __init__(
        self,
        net_loss,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell_PACS, self).__init__(auto_prefix=auto_prefix)
        self.net_loss = net_loss
        self.net_loss.set_grad()
        # self.CEnet.add_flags(defer_inline=True)

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x, y):
        loss = self.net_loss(x, y)
        # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net_loss, self.weights)(x, y, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_DANN(nn.Cell):
    def __init__(
        self,
        net_loss,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell_DANN, self).__init__(auto_prefix=auto_prefix)
        self.net_loss = net_loss
        self.net_loss.set_grad()
        # self.CEnet.add_flags(defer_inline=True)

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x1, x2, y1, y2):
        loss = self.net_loss(x1, x2, y1, y2)
        # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net_loss, self.weights)(x1, x2, y1, y2, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_MV3(nn.Cell):
    def __init__(
        self,
        net_loss,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell_MV3, self).__init__(auto_prefix=auto_prefix)
        self.net_loss = net_loss
        self.net_loss.set_grad()
        # self.CEnet.add_flags(defer_inline=True)

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x1, x2, y1, y2):
        loss = self.net_loss(x1, x2, y1, y2)
        # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net_loss, self.weights)(x1, x2, y1, y2, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss