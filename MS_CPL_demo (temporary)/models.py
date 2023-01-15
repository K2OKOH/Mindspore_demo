# noinspection PyUnresolvedReferences
import mindspore
from mindspore import nn
from mindspore.common.initializer import initializer, XavierUniform, Zero

rdc_text_dim = 1000
z_dim = 100
h_dim = 1024

def weights_init():
    return XavierUniform()

def bias_init():
    return Zero()

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

