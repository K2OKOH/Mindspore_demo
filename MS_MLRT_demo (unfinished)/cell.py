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

class TrainOneStepCell_MLRT(nn.Cell):
    def __init__(
        self,
        net,
        loss,
        optimizer: nn.Optimizer,
        sens=1.0,
        auto_prefix=True,
    ):
        super(TrainOneStepCell_MLRT, self).__init__(auto_prefix=auto_prefix)
        # 定义网络
        self.m_train_net = net
        self.m_test_net = net
        self.ce_loss = loss

        # self.CEnet.add_flags(defer_inline=True)
        # 获取参数
        self.m_train_params = mindspore.ParameterTuple(self.m_train_net.trainable_params())
        self.m_test_params = mindspore.ParameterTuple(self.m_test_net.trainable_params())
        
        # self.moving_m_train_params = mindspore.ParameterTuple(self.m_train_net.untrainable_params())
        # self.moving_m_test_params = mindspore.ParameterTuple(self.m_test_net.untrainable_params())
        # self.m_train_params_trainable = mindspore.ParameterTuple(self.m_train_net.trainable_params())

        # 网络加上loss
        self.m_train_net_withloss = nn.WithLossCell(self.m_train_net, self.ce_loss)
        self.m_test_net_withloss = nn.WithLossCell(self.m_test_net, self.ce_loss)
        # self.m_test_net_withloss = net_loss

        self.m_train_net_withloss.set_grad()
        self.m_test_net_withloss.set_grad()

        # 优化器
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters

        self.grad = C.GradOperation(get_by_list=True, sens_param=True)

        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = F.identity

        # self.outer_params = mindspore.ParameterTuple(self.meta_train_net_loss.trainable_params())
        # self.inner_params = mindspore.ParameterTuple(self.inner_net.trainable_params())

    def construct(self, x1, x2, y1, y2):
        # 复制参数
        self.assign_tuple(self.m_train_params, self.m_test_params)
        
        loss_train = self.m_train_net_withloss(x1, y1)
        # # 反向传播的缩放系数
        sens = P.Fill()(P.DType()(loss_train), P.Shape()(loss_train), self.sens)
        grad_train = self.grad(self.m_train_net_withloss, self.m_train_params)(x1, y1, sens)
        grad_train = self.grad_reducer(grad_train)
        fast_weights = self.get_fast_weights(self.m_train_params, grad_train, 0.0001)
        self.assign_tuple(self.m_test_params, fast_weights)

        loss_test = self.m_test_net_withloss(x2, y2)
        sens = P.Fill()(P.DType()(loss_test), P.Shape()(loss_test), self.sens)
        grad_test = self.grad(self.m_test_net_withloss, self.weights)(x2, y2, sens)
        grad_test = self.grad_reducer(grad_test)
        # fast_weights = self.get_fast_weights(self.m_test_params, grad, 0.003)
        loss = loss_train + loss_test

        loss = F.depend(loss, self.optimizer(grad_test))
        return loss

    def assign_tuple(self, param, value):
        """assign params from tuple(value) to tuple(param)"""
        for i in range(len(param)):
            ops.assign(param[i], value[i])
        return param

    def get_fast_weights(self, param, grad, lr):
        """fast_weights = param - lr * grad"""
        fast_weight = []
        for i in range(len(param)):
            fast_weight.append(param[i] - lr * grad[i])
        return fast_weight