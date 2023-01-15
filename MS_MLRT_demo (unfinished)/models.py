import mindspore
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops

class Generator(nn.Cell):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, pad_mode='valid')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, pad_mode='valid')
        self.bn2 = nn.BatchNorm2d(48)
        self.max_pool = nn.MaxPool2d(stride=2, kernel_size=2)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)
        self.mean = ms.ops.ReduceMean(keep_dims=True)

    def construct(self, x):
        x = self.mean(x, 1)
        x = x.view(x.shape[0],1,x.shape[2],x.shape[3])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = x.view(32, 768)
        return x

class GradReverse(nn.Cell):
    def __init__(self, lambd):
        self.lambd = lambd

    def construct(self, x):
        return x.view(x)

    def bprop(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)

class Classifier(nn.Cell):
    def __init__(self, prob=0.5):
        super(Classifier, self).__init__()
        self.fc1 = nn.Dense(48*4*4, 100)
        self.bn1_fc = nn.BatchNorm1d(100)
        self.fc2 = nn.Dense(100, 100)
        self.bn2_fc = nn.BatchNorm1d(100)
        self.fc3 = nn.Dense(100, 10)
        self.bn_fc3 = nn.BatchNorm1d(10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def construct(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.dropout(x)
        x = self.relu(self.bn1_fc(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2_fc(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ad_layer1 = nn.Dense(48*4*4, 1024)
        self.ad_layer2 = nn.Dense(1024,1024)
        self.ad_layer3 = nn.Dense(1024, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = ops.Sigmoid()
    def construct(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x

class Mixer(nn.Cell):
    def __init__(self):
        super(Mixer, self).__init__()
        self.fc1_x = nn.Dense(768,512)
        self.fc2 = nn.Dense(512,1)
        self.relu = nn.ReLU()
    def construct(self, x):
        x = self.relu(self.fc1_x(x))
        x = self.fc2(x)
        return x

# 组合网络
class G14_C_net(nn.Cell):
    def __init__(self, G1_net, G4_net, C_net):
        super(G14_C_net, self).__init__()
        self.g1 = G1_net
        self.g4 = G4_net
        self.c = C_net

    def construct(self, x):
        x = self.g1(x) + self.g4(x)
        x = self.c(x)
        return x

class G1_D_net(nn.Cell):
    def __init__(self, G1_net, D_net):
        super(G1_D_net, self).__init__()
        self.g1 = G1_net
        self.d = D_net

    def construct(self, x):
        x = self.g1(x)
        x = self.d(x)
        return x

class G12_net(nn.Cell):
    def __init__(self, G1_net, G2_net):
        super(G12_net, self).__init__()
        self.g1 = G1_net
        self.g2 = G2_net

    def construct(self, x_s, x_t):
        feat_g1s = self.g1(x_s)
        feat_g1t = self.g1(x_t)
        feat_g2s = self.g2(x_s)
        feat_g2t = self.g2(x_t)
        return feat_g1s, feat_g1t, feat_g2s, feat_g2t

class G3_C_net(nn.Cell):
    def __init__(self, G3_net, Cls_net):
        super(G3_C_net, self).__init__()
        self.g3 = G3_net
        self.c = Cls_net

    def construct(self, x):
        x = self.g3(x)
        x = self.c(x)
        return x

class G4_Cs_net(nn.Cell):
    def __init__(self, G4_net, Cls_net, Cls1_net, Cls2_net):
        super(G4_Cs_net, self).__init__()
        self.g4 = G4_net
        self.c = Cls_net
        self.c1 = Cls1_net
        self.c2 = Cls2_net

    def construct(self, x_s, x_t):
        feat_s = self.g4(x_s)
        feat_t = self.g4(x_t)

        feat_s_C = self.c(feat_s)
        feat_t_C = self.c(feat_t)
        feat_s_C1 = self.c1(feat_s)
        feat_s_C2 = self.c2(feat_s) 
        feat_t_C1 = self.c1(feat_t)
        feat_t_C2 = self.c2(feat_t)

        return feat_s_C, feat_t_C, feat_s_C1, feat_s_C2, feat_t_C1, feat_t_C2

class G1234_M_net(nn.Cell):
    def __init__(self, G1_net, G2_net, G3_net, G4_net, Mix):
        super(G1234_M_net, self).__init__()
        self.g1 = G1_net
        self.g2 = G2_net
        self.g3 = G3_net
        self.g4 = G4_net
        self.m = Mix

    def construct(self, x_t):
        feat_1 = self.g1(x_t)
        feat_2 = self.g2(x_t)
        feat_3 = self.g3(x_t)
        feat_4 = self.g4(x_t)

        mix_1 = self.m(feat_1)
        mix_2 = self.m(feat_2)
        mix_3 = self.m(feat_3)
        mix_4 = self.m(feat_4)

        return mix_1, mix_2, mix_3, mix_4

class Ite4_net(nn.Cell):
    def __init__(self, G1_net, G2_net, G3_net, G4_net, Cls_net, Cls1_net, Cls2_net, D_net, Mix):
        super(Ite4_net, self).__init__()
        self.g1 = G1_net
        self.g2 = G2_net
        self.g3 = G3_net
        self.g4 = G4_net
        self.c = Cls_net
        self.c1 = Cls1_net
        self.c2 = Cls2_net
        self.d = D_net
        self.m = Mix

    def construct(self, x_s, x_t):
        feat_s4 = self.g4(x_s)
        feat_t4 = self.g4(x_t)
        output_s_D = self.d(feat_s4)
        output_t_D = self.d(feat_t4)
        feat_t4 = self.g4(x_t)

        output_t_C = self.c(feat_t4)
        output_t_C1 = self.c1(feat_t4)
        output_t_C2 = self.c2(feat_t4)

        feat_1 = self.g1(x_t)
        feat_2 = self.g2(x_t)
        feat_3 = self.g3(x_t)
        feat_4 = self.g4(x_t)

        mix_1 = self.m(feat_1)
        mix_2 = self.m(feat_2)
        mix_3 = self.m(feat_3)
        mix_4 = self.m(feat_4)

        return mix_1, mix_2, mix_3, mix_4, output_s_D, output_t_D, output_t_C, output_t_C1, output_t_C2
