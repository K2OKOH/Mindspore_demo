import mindspore
import mindspore as ms
from mindspore import nn
import mindspore.ops as ops

class GradReverse(nn.Cell):
    def __init__(self, lambd):
        super(GradReverse, self).__init__()
        self.lambd = lambd

    def construct(self, x):
        return x

    def bprop(self, x, out, grad_output):
        return (grad_output * -self.lambd)

class AlexNet(nn.Cell):
    def __init__(self, num_classes=7):
        super(AlexNet, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
            
        )

    def construct(self, x):
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        class_out = self.classifier(feat)
        return class_out

class AlexNet_DANN(nn.Cell):
    def __init__(self, num_classes=7):
        super(AlexNet_DANN, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
            
        )
        self.doamin_classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, 2),
        )
        self.grl = GradReverse(1)

    def construct(self, x1):
        feat_s1 = self.features(x1)
        feat_s1 = feat_s1.view(feat_s1.shape[0], -1)
        # label classifier
        class_out_s1 = self.classifier(feat_s1)
        # domain classifier
        feat_s1 = self.grl(feat_s1)
        domain_out_s1 = self.doamin_classifier(feat_s1)

        return class_out_s1, domain_out_s1

class AlexNet_MV3(nn.Cell):
    def __init__(self, num_classes=7):
        super(AlexNet_MV3, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, pad_mode='pad'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dense(4096, num_classes),
        )
        self.encoder_1 = nn.SequentialCell(
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dense(4096, 2048),
        )
        self.encoder_2 = nn.SequentialCell(
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dense(4096, 2048),
        )
        self.encoder_3 = nn.SequentialCell(
            nn.Dense(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dense(4096, 2048),
        )
        self.decoder_1 = nn.SequentialCell(
            nn.Dense(2048, 4096),
            nn.ReLU(),
            nn.Dense(4096, 256 * 6 * 6),
        )
        self.decoder_2 = nn.SequentialCell(
            nn.Dense(2048, 4096),
            nn.ReLU(),
            nn.Dense(4096, 256 * 6 * 6),
        )
        self.decoder_3 = nn.SequentialCell(
            nn.Dense(2048, 4096),
            nn.ReLU(),
            nn.Dense(4096, 256 * 6 * 6),
        )
        self.doamin_1 = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(512, 2),
        )
        self.doamin_2 = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(512, 2),
        )
        self.doamin_3 = nn.SequentialCell(
            nn.Dropout(),
            nn.Dense(2048, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(512, 2),
        )
        self.grl = GradReverse(0.2)

    def construct(self, x):
        feat = self.features(x)
        feat = feat.view(feat.shape[0], -1)
        # label classifier
        class_out = self.classifier(feat)
        # View 1
        cp1 = self.encoder_1(feat)
        rc1 = self.decoder_1(cp1)
        re_cp1 = self.grl(cp1)
        domain_1_out = self.doamin_1(re_cp1)
        # View 2
        cp2 = self.encoder_2(feat)
        rc2 = self.decoder_2(cp2)
        re_cp2 = self.grl(cp2)
        domain_2_out = self.doamin_2(re_cp2)
        # View 3
        cp3 = self.encoder_3(feat)
        rc3 = self.decoder_3(cp3)
        re_cp3 = self.grl(cp3)
        domain_3_out = self.doamin_3(re_cp3)

        return class_out, feat, rc1, rc2, rc3, \
            cp1, cp2, cp3, \
            domain_1_out, domain_2_out, domain_3_out
