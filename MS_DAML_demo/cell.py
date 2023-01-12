# noinspection PyUnresolvedReferences
import mindspore
from mindspore import nn
import mindspore.ops.operations as P
import mindspore.ops.functional as F
import mindspore.ops.composite as C

from mindspore.ops import ExpandDims, Argmax
import mindspore.numpy as mnp
from mindspore import ops

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

class G1_net_loss(nn.Cell):
    def __init__(self, G1_net, loss_fn, auto_prefix=False):
        super(G1_net_loss, self).__init__(auto_prefix=False)
        self.net = G1_net
        self.loss_fn = loss_fn

    def construct(self, s, t):
        feat_s = self.net(s)
        feat_t = self.net(t)
        loss = self.loss_fn(feat_s, feat_t)*0.2
        return loss

class MeanSoftmax(nn.Cell):
    def __init__(self, auto_prefix=False):
        super(MeanSoftmax, self).__init__(auto_prefix=False)
        self.mean = ops.ReduceMean()
        self.softmax = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, x1, x2): 
        loss = self.mean(self.abs(self.softmax(x1) - self.softmax(x2)))
        return loss

class G12_net_loss(nn.Cell):
    def __init__(self, G12_net, loss_fn, auto_prefix=False):
        super(G12_net_loss, self).__init__(auto_prefix=False)
        self.net = G12_net
        self.loss_fn = loss_fn

    def construct(self, s, t):
        feat_g1s, feat_g1t, feat_g2s, feat_g2t = self.net(s, t)
        loss = self.loss_fn(feat_g1s, feat_g2s) + self.loss_fn(feat_g1t, feat_g2t) + self.loss_fn(feat_g2s, feat_g2t)
        return -loss

class G3_C_net_loss(nn.Cell):
    def __init__(self, G3_C_net, auto_prefix=False):
        super(G3_C_net_loss, self).__init__(auto_prefix=False)
        self.net = G3_C_net
        self.exp = ops.Exp()
        self.log = ops.Log()

    def construct(self, x):
        x = self.net(x)
        x = self.exp(x)
        loss_G3_C = (1/class_num) * self.log(x)
        return loss_G3_C

class G4_Cs_net_loss(nn.Cell):
    def __init__(self, G3_C_net, crit, auto_prefix=False):
        super(G4_Cs_net_loss, self).__init__(auto_prefix=False)
        self.net = G3_C_net
        self.loss_fn = crit
        self.mean = ops.ReduceMean()
        self.softmax = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, x_s, x_t, y):
        feat_s_C, feat_t_C, feat_s_C1, feat_s_C2, feat_t_C1, feat_t_C2 = self.net(x_s, x_t)
        loss_s = self.loss_fn(feat_s_C1, y) + self.loss_fn(feat_s_C2, y) + self.loss_fn(feat_s_C, y)
        loss_t = self.mean(self.abs(self.softmax(feat_t_C1) - self.softmax(feat_t_C2)))
        return loss_s + loss_t

class G1234_M_net_loss(nn.Cell):
    def __init__(self, G1234_M_net, auto_prefix=False):
        super(G1234_M_net_loss, self).__init__(auto_prefix=False)
        self.net = G1234_M_net
        self.mean = ops.ReduceMean()
        self.softmax = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, x_t):
        mix_1, mix_2, mix_3, mix_4 = self.net(x_t)
        l_13 = - self.mean(self.abs(self.softmax(mix_1) - self.softmax(mix_3)))
        l_14 = self.mean(self.abs(self.softmax(mix_1) - self.softmax(mix_4)))
        l_23 = self.mean(self.abs(self.softmax(mix_2) - self.softmax(mix_3)))
        l_24 = - self.mean(self.abs(self.softmax(mix_2) - self.softmax(mix_4)))
        
        loss_mal = -(l_13 + l_14 + l_23 + l_24)

        return loss_mal

class I4_net_loss(nn.Cell):
    def __init__(self, I4_net, loss_fn, auto_prefix=False):
        super(I4_net_loss, self).__init__(auto_prefix=False)
        self.net = I4_net
        self.loss_fn = loss_fn
        self.mean = ops.ReduceMean()
        self.softmax = ops.Sigmoid()
        self.abs = ops.Abs()

    def construct(self, x_s, x_t):
        mix_1, mix_2, mix_3, mix_4, output_s_D, output_t_D, output_t_C, output_t_C1, output_t_C2 = self.net(x_s, x_t)
        loss_bce1 = - self.loss_fn(output_s_D , output_t_D)
        loss_6 = 0.2*loss_bce1

        loss_71 = self.mean(self.abs(self.softmax(output_t_C1) - self.softmax(output_t_C2)))
        loss_72 = self.mean(self.abs(self.softmax(output_t_C) - self.softmax(output_t_C1)))
        loss_73 = self.mean(self.abs(self.softmax(output_t_C) - self.softmax(output_t_C2)))
        loss_7 = loss_71 + loss_72 + loss_73

        l_13 = - self.mean(self.abs(self.softmax(mix_1) - self.softmax(mix_3)))
        l_14 = self.mean(self.abs(self.softmax(mix_1) - self.softmax(mix_4)))
        l_23 = self.mean(self.abs(self.softmax(mix_2) - self.softmax(mix_3)))
        l_24 = - self.mean(self.abs(self.softmax(mix_2) - self.softmax(mix_4)))
        
        loss_mal = -(l_13 + l_14 + l_23 + l_24)

        loss_all = loss_6 + loss_7 + loss_mal

        return loss_all

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
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.CEnet, self.weights)(x, y, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_G1_BCE(nn.Cell):
    def __init__(
        self,
        net,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell_G1_BCE, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x_s, x_t):
        loss = self.net(x_s, x_t)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x_s, x_t, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_G12_MS(nn.Cell):
    def __init__(self, net, optimizer: nn.Optimizer, sens=1.0, auto_prefix=True):
        super(TrainOneStepCell_G12_MS, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x_s, x_t):
        loss = self.net(x_s, x_t)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x_s, x_t, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_G3_C_Loss(nn.Cell):
    def __init__(self, net, optimizer: nn.Optimizer, sens=1.0, auto_prefix=True):
        super(TrainOneStepCell_G3_C_Loss, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x):
        loss = self.net(x)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_G4_Cs_Loss(nn.Cell):
    def __init__(self, net, optimizer: nn.Optimizer, sens=1.0, auto_prefix=True):
        super(TrainOneStepCell_G4_Cs_Loss, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x_s, x_t, y):
        loss = self.net(x_s, x_t, y)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x_s, x_t, y, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_G1234_M_Loss(nn.Cell):
    def __init__(self, net, optimizer: nn.Optimizer, sens=1.0, auto_prefix=True):
        super(TrainOneStepCell_G1234_M_Loss, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x_t):
        loss = self.net(x_t)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x_t, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss

class TrainOneStepCell_I4_Loss(nn.Cell):
    def __init__(self, net, optimizer: nn.Optimizer, sens=1.0, auto_prefix=True):
        super(TrainOneStepCell_I4_Loss, self).__init__(auto_prefix=auto_prefix)
        self.net = net
        self.net.set_grad()

        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

    def construct(self, x_s, x_t):
        loss = self.net(x_s, x_t)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.net, self.weights)(x_s, x_t, sens)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
